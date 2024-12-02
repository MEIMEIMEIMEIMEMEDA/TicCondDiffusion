from typing import Dict, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mapsa.data.data_types import DiffusionLMOutput
from mapsa.data.data_types import MetricType
from mapsa.tools.matcher import HungarianMatcher

def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

# 导入用于计算 KL 散度和离散高斯对数似然的函数
def normal_kl(mean1, logvar1, mean2, logvar2):
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 60.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 60.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

class LossInput(NamedTuple):
  pred: torch.Tensor = None
  tgt_token_ids: torch.Tensor = None
  tgt_mask: torch.Tensor = None


class DiffusionLMLossInput(NamedTuple):
  pred: DiffusionLMOutput = None
  gt_types: torch.Tensor = None
  gt_spans: torch.Tensor = None
  token_masks: torch.Tensor = None
  true_log_variance_clipped: torch.Tensor = None
  true_mean: torch.Tensor = None
  pre_mean: torch.Tensor = None
  pre_x_start: torch.Tensor = None
  t: torch.Tensor = None


class SpanLoss(nn.Module):

  def __init__(self):
    super(SpanLoss, self).__init__()

  def forward(self, loss: LossInput):
    pred, tgt, tgt_mask = loss.pred, loss.tgt_token_ids, loss.tgt_mask
    tgt = tgt.masked_fill(tgt_mask.eq(0), -100) if tgt_mask is not None else tgt
    return F.cross_entropy(
        input=pred.transpose(1, 2),
        target=tgt,
    )


class AspectDETRCriterion(nn.Module):
  """This class computes the loss for DETR using a diffusion algorithm
  approach."""

  def __init__(
      self,
      aspect_type_count,
      weight_dict,
      nil_weight,
      losses,
      type_loss,
      match_class_weight,
      match_boundary_weight,
      match_boundary_type,
      solver,
  ):
    """Initialize the criterion.

    Parameters:
        aspect_type_count: Number of aspect types.
        weight_dict: Dictionary with loss weights.
        nil_weight: Weight for the nil class.
        losses: List of loss types to be applied.
        type_loss: Type of loss ('celoss' or 'bceloss').
        match_class_weight: Weight for class matching.
        match_boundary_weight: Weight for boundary matching.
        match_boundary_type: Type of boundary matching.
        solver: Solver for the matcher.
    """
    super().__init__()
    self.aspect_type_count = aspect_type_count
    self.matcher = HungarianMatcher(
        cost_class=match_class_weight,
        cost_span=match_boundary_weight,
        match_boundary_type=match_boundary_type,
        solver=solver,
    )
    self.weight_dict = weight_dict
    self.nil_weight = nil_weight
    self.losses = losses
    empty_weight = torch.ones(self.aspect_type_count)
    empty_weight[0] = self.nil_weight
    self.register_buffer('empty_weight', empty_weight)
    self.type_loss = type_loss

  def loss_labels(
      self,
      outputs: Dict[str, torch.Tensor],
      targets: Dict[str, torch.Tensor],
      indices,
      num_spans,
  ):
    """Compute the classification loss (NLL)."""
    assert 'pred_logits' in outputs

    src_logits = outputs['pred_logits']  # torch.Size([32, 60, 4])
    idx = self._get_src_permutation_idx(indices)
    labels = targets['labels']

    target_classes_o = torch.cat([t[J] for t, (_, J) in zip(labels, indices)])
    target_classes = torch.full(src_logits.shape[:2],
                                0,
                                dtype=torch.int64,
                                device=src_logits.device)
    target_classes[idx] = target_classes_o  # torch.Size([32, 60])
    # empty_weight = self.empty_weight.clone()

    # if self.nil_weight == -1:
    #     empty_weight[0] = num_spans / (
    #         src_logits.size(0) * src_logits.size(1) - num_spans
    #     )
    if self.type_loss == 'celoss':
      loss_ce = self._compute_ce_loss(src_logits, target_classes,
                                      self.empty_weight)
    elif self.type_loss == 'bceloss':
      loss_ce = self._compute_bce_loss(src_logits, target_classes)

    return {'loss_ce': loss_ce.mean()}

  def _compute_ce_loss(self, src_logits, target_classes, empty_weight):
    src_logits = src_logits.view(-1,
                                 src_logits.size(2))  # torch.Size([1920, 4])
    target_classes = target_classes.view(-1)  # 1920
    return F.cross_entropy(src_logits,
                           target_classes,
                           empty_weight,
                           reduction='none')

  def _compute_bce_loss(self, src_logits, target_classes):
    src_logits = src_logits.view(-1, src_logits.size(2))
    target_classes = target_classes.view(-1)
    target_classes_onehot = torch.zeros(
        [target_classes.size(0), src_logits.size(1)],
        dtype=torch.float32).to(device=target_classes.device)
    target_classes_onehot.scatter_(1, target_classes.unsqueeze(1), 1)
    src_logits_p = torch.sigmoid(src_logits)
    return F.binary_cross_entropy(src_logits_p,
                                  target_classes_onehot,
                                  reduction='none')

  def loss_boundary(self, outputs, targets, indices, num_spans):
    """Compute the boundary loss."""
    idx = self._get_src_permutation_idx(indices)
    # torch.Size([1920, 80])
    src_spans_left, src_spans_right = (
        outputs['pred_left'][idx],
        outputs['pred_right'][idx],
    )
    token_masks = (outputs['token_mask'].unsqueeze(1).expand(
        -1, outputs['pred_right'].size(1), -1))
    token_masks = token_masks[idx]
    # 32*60
    gt_left = targets['gt_left']
    # 1920
    target_spans_left = torch.cat([t[i] for t, (_, i) in zip(gt_left, indices)],
                                  dim=0)
    gt_right = targets['gt_right']
    target_spans_right = torch.cat(
        [t[i] for t, (_, i) in zip(gt_right, indices)], dim=0)
    # src_spans 1920*80, target_spans 1920
    left_nll_loss = self._compute_boundary_loss(src_spans_left,
                                                target_spans_left)
    right_nll_loss = self._compute_boundary_loss(src_spans_right,
                                                 target_spans_right)

    loss_boundary = (left_nll_loss + right_nll_loss) * token_masks
    return {'loss_boundary': loss_boundary.sum() / int(num_spans)}

  def _compute_boundary_loss(
      self, src_spans, target_spans):  # src_spans 1920*80, target_spans 1920
    onehot = torch.zeros(
        [target_spans.size(0), src_spans.size(1)],
        dtype=torch.float32).to(device=target_spans.device)  #1920*80
    onehot.scatter_(1, target_spans.unsqueeze(1), 1)
    return F.binary_cross_entropy(src_spans, onehot, reduction='none')

  def _get_src_permutation_idx(self, indices):
    """Permute predictions following indices."""
    batch_idx = torch.cat(
        [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

  def _get_tgt_permutation_idx(self, indices):
    """Permute targets following indices."""
    batch_idx = torch.cat(
        [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx
  
  def loss_variance_and_mean(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        indices,
        num_spans,
    ):
        """Compute the variance and mean loss."""
        """Compute the variance and mean loss."""
        idx = self._get_src_permutation_idx(indices)

        pred_left_log_variance = outputs['pred_left_log_variance']  # torch.Size([32, 60, 60])
        pred_right_log_variance = outputs['pred_right_log_variance']  # torch.Size([32, 60, 60])
        pred_logits_log_variance = outputs['pred_logits_log_variance']  # torch.Size([32, 60])

        # 对于 pred_left_log_variance 和 pred_right_log_variance，我们将沿着最后一个维度取平均
        pred_left_log_variance = pred_left_log_variance.mean(dim=2, keepdim=True)  # 结果维度将为 [32, 60, 1]
        pred_right_log_variance = pred_right_log_variance.mean(dim=2, keepdim=True)  # 结果维度将为 [32, 60, 1]

        # 将两个张量沿着最后一个维度拼接
        pred_left_right_log_variance = torch.cat([pred_left_log_variance, pred_right_log_variance], dim=2)  # 结果维度将为 [32, 60, 2]

        # 然后复制最后一个维度，使其成为 [32, 60, 2]
        pred_logits_log_variance = pred_logits_log_variance.unsqueeze(-1)
        pred_logits_log_variance = pred_logits_log_variance.expand_as(pred_left_right_log_variance)

        # 相加，[32, 60, 2]
        pred_log_variance = pred_left_right_log_variance #+ pred_logits_log_variance

        # true_log_variance，[32, 1, 1] -> [32, 60, 2]
        pred_size = pred_log_variance.size()
        true_log_variance_clipped = targets['true_log_variance_clipped']
        true_log_variance_clipped = true_log_variance_clipped.expand(*pred_size)

        true_mean = targets['true_mean']  # torch.Size([32, 60, 2])
        pred_mean = outputs['pred_mean']  # torch.Size([32, 60, 2])

        # Compute KL divergence loss 
        kl_loss = normal_kl(true_mean, true_log_variance_clipped, pred_mean, pred_log_variance)
        kl_loss = mean_flat(kl_loss) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            outputs['pred_x_start'], means=pred_mean, log_scales=0.5 * pred_log_variance
        )
        assert decoder_nll.shape == outputs['pred_x_start'].shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((outputs['t'] == 0), decoder_nll, kl_loss)
        return {'loss_variance_and_mean': output.mean()}  # 返回一个字典
  
        # idx = self._get_src_permutation_idx(indices)
        # print(targets['true_mean'][idx].shape)
        # pred_left_log_variance = outputs['pred_left_log_variance'][idx]  #  pred_left_log_variance : torch.Size([1920, 60]), outputs['pred_left_log_variance']: torch.Size([32, 60, 60])
        # pred_right_log_variance = outputs['pred_right_log_variance'][idx] # torch.Size([1920, 60])
        # pred_logits_log_variance = outputs['pred_logits_log_variance'][idx] # torch.Size([1920])

        # # 对于 pred_left_log_variance 和 pred_right_log_variance，我们将沿着最后一个维度取平均
        # pred_left_log_variance = pred_left_log_variance.mean(dim=1, keepdim=True)  # 结果维度将为 [1920, 1]
        # pred_right_log_variance = pred_right_log_variance.mean(dim=1, keepdim=True) # 结果维度将为 [1920, 1]

        # # 将两个张量沿着最后一个维度拼接
        # pred_left_right_log_variance = torch.cat([pred_left_log_variance, pred_right_log_variance], dim=1)  # 结果维度将为 [1920, 2]

        # # 然后复制最后一个维度，使其成为 [1920, 2]
        # pred_logits_log_variance = pred_logits_log_variance.unsqueeze(-1)
        # pred_logits_log_variance = pred_logits_log_variance.repeat(1, 2)

        # # 相加,[1920, 2]
        # pred_log_variance = pred_left_right_log_variance + pred_logits_log_variance

        # # true_log_variance，[1920，1] [1920, 2]
        # true_log_variance_clipped = targets['true_log_variance_clipped'][idx] 
        # # 变换 true_log_variance_clipped 的视图
        # true_log_variance_clipped = torch.cat((true_log_variance_clipped, true_log_variance_clipped), dim=1)
        
      
        # true_mean = targets['true_mean'][idx]  # [1920, 2]
        # pred_mean = outputs['pred_mean'][idx]

        # # Compute KL divergence loss 
        # kl_loss = normal_kl(true_mean, true_log_variance_clipped, pred_mean, pred_log_variance)
        # kl_loss = mean_flat(kl_loss) / np.log(2.0)
  
        # decoder_nll = -discretized_gaussian_log_likelihood(
        #     outputs['pred_x_start'].view(pred_log_size)[idx], means=pred_mean, log_scales=0.5 * pred_log_variance
        # )
        # assert decoder_nll.shape == outputs['pred_x_start'].shape
        # decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # # At the first timestep return the decoder NLL,
        # # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        # output = torch.where((outputs['t'] == 0), decoder_nll, kl_loss)
        # return {'loss_variance_and_mean': output.mean()}

    

  def get_loss(self, loss, outputs, targets, indices, num_spans, **kwargs):
    """Get the specified loss."""
    loss_map = {
        'labels': self.loss_labels,
        'boundary': self.loss_boundary,
        'variance_and_mean': self.loss_variance_and_mean,  # 添加方差和均值的损失计算
    }
    assert loss in loss_map, f'Do you really want to compute {loss} loss?'
    return loss_map[loss](outputs, targets, indices, num_spans, **kwargs)


  def forward(self, outputs, targets, indices=None):
    """Compute all the requested losses."""
    if indices is None:
      indices = self.matcher(outputs, targets)

    num_spans = sum(targets['sizes'])

    losses = {}
    for loss in self.losses:
      losses.update(self.get_loss(loss, outputs, targets, indices, num_spans))
    return losses, indices


class ABSADiffusionLoss(nn.Module):
  """Class for computing the ABSADiffusionLoss."""

  def __init__(
      self,
      aspect_type_count=4,
      nil_weight=-1.0,
      match_class_weight=1.0,
      match_boundary_weight=1.0,
      loss_class_weight=1.0,
      loss_boundary_weight=1.0,
      loss_variance_and_mean_weight=1.0,
      match_boundary_type='logp',
      type_loss='celoss',
      solver='hungarian',
  ):
    super().__init__()

    self.weight_dict = {
        'loss_ce': loss_class_weight,
        'loss_boundary': loss_boundary_weight,
        'loss_variance_and_mean': loss_variance_and_mean_weight,  # 添加方差和均值的损失权重
    }
    losses = ['labels', 'boundary', 'variance_and_mean']
    self.criterion = AspectDETRCriterion(
        aspect_type_count,
        self.weight_dict,
        nil_weight,
        losses,
        type_loss=type_loss,
        match_class_weight=match_class_weight,
        match_boundary_weight=match_boundary_weight,
        match_boundary_type=match_boundary_type,
        solver=solver,
    )

  def to(self, device):
    super().to(device)
    self.criterion.to(device)

  def forward(self, loss_input: DiffusionLMLossInput):
    gt_types, gt_spans, gt_var, gt_mean= (
        loss_input.gt_types,
        loss_input.gt_spans,
        loss_input.true_log_variance_clipped,
        loss_input.true_mean,
    )

    if len(gt_types) == 0:
      return torch.tensor(0.1, device=loss_input.pred.left_boundary.device)

    sizes = [s.shape[0] for s in gt_spans]

    targets = {
        'labels': gt_types,
        'gt_left': gt_spans[:, :, 0],
        'gt_right': gt_spans[:, :, 1],
        'sizes': sizes,
        'true_log_variance_clipped': gt_var,
        'true_mean': gt_mean,
    }

    outputs = {
        'pred_logits': loss_input.pred.cls_logits,
        'pred_left': loss_input.pred.left_boundary,
        'pred_right': loss_input.pred.right_boundary,
        'token_mask': loss_input.token_masks,
        'pred_left_log_variance': loss_input.pred.left_variance,
        'pred_right_log_variance': loss_input.pred.right_variance,
        'pred_logits_log_variance': loss_input.pred.logits_variance,
        'pred_mean': loss_input.pre_mean,
        'pred_x_start': loss_input.pre_x_start,
        't': loss_input.t,
    }

    loss_dict, indices = self.criterion(outputs, targets)

    train_loss = sum(
        loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys())
    return train_loss


def get_loss(metric_type: MetricType = MetricType.CLASSIFICATION):
  if isinstance(metric_type, str):
    metric_type = MetricType[metric_type.upper()]
  if metric_type == MetricType.CLASSIFICATION:
    return nn.CrossEntropyLoss()
  if metric_type == MetricType.SPAN:
    return SpanLoss()
  if metric_type == MetricType.ABSA_DIFFUSION:
    return ABSADiffusionLoss
  raise ValueError(f'Please check your input {list(MetricType)}')
