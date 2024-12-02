import copy
import logging
import math
import random
from typing import List, Tuple

import torch
from torch import nn as nn
from torch.nn import functional as F
from transformers import PretrainedConfig
from transformers import PreTrainedModel

from mapsa.data.data_types import DiffusionLMOutput
from mapsa.data.data_types import DiffusionTargets
from mapsa.model.model_wrapper import AutoModelWrapper
from mapsa.tools.loss import DiffusionLMLossInput
from mapsa.tools.loss import get_loss

logger = logging.getLogger()

class MAPSADiffusion(PreTrainedModel):

    def __init__(self, config: PretrainedConfig, **kwargs):
        super().__init__(config)

        self.text_config = self.config
        if hasattr(self.config, "text_config"):
            self.text_config = self.config.text_config

        self.cls_nums = kwargs.get("cls_nums", 4)
        self.pool_type = kwargs.get("pool_type", "max")
        self.caption_span_attn_layers = kwargs.get("caption_span_attn_layers", 1)
        self.span_attn_layers = kwargs.get("span_attn_layers", 2)
        self.image_guidance_layers = kwargs.get("image_guidance_layers", 1)
        self.wo_self_attn = kwargs.get("wo_self_attn", False)
        self.wo_cross_attn = kwargs.get("wo_cross_attn", False)
        self.soi_pooling = kwargs.get("soi_pooling", "sumpool+lrconcat")
        self.pos_type = kwargs.get("pos_type", "sine")
        self.step_embed_type = kwargs.get("step_embed_type", "scaleshift")
        self.sample_dist_type = kwargs.get("sample_dist_type", "normal")
        self.num_proposals = kwargs.get("num_proposals", 80)
        self.beta_schedule = kwargs.get("beta_schedule", "cosine")
        self.extand_noise_spans = kwargs.get("extand_noise_spans", "concat")
        self.timesteps = kwargs.get("timesteps", 1000)
        self.sampling_timesteps = kwargs.get("sampling_timesteps", 5)
        self.scale = kwargs.get("scale", 1.0)
        self.span_renewal = kwargs.get("span_renewal", False)
        self.step_ensemble = kwargs.get("step_ensemble", False)
        self.ddim_sampling_eta = 1.0
        self.p2_loss_weight_gamma = 0.0
        self.fp2_loss_weight_k = 1
        self.dropout = kwargs.get("prop_drop", 0.2)
        self.loss_fn = get_loss("ABSA_DIFFUSION")()
        self.pos_embeddings = None

        self._init_encoder(config)
        self._init_pooling_layers()
        self._init_pos_embeddings()
        self._init_image_guided_blocks()
        self._init_span_attention_layers()
        self._init_predictors()
        self._init_step_embedding()
        self._init_se_blocks()
        self._init_LSTM()
        self._init_caption_span_attention_layers()
        self._init_aspect_span_embedding()
        self.init_weights()
        self.build_diffusion()
      
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor = None,
        labels: torch.Tensor = None,
        image_caption_token_ids: torch.Tensor = None,
        image_caption_att_mask: torch.Tensor = None,
        **kwargs,
    ) -> DiffusionLMOutput:
        image_caption_token_ids = image_caption_token_ids.squeeze(1)  # Removes the second dimension
        image_caption_att_mask = image_caption_att_mask.squeeze(1)  # Removes the second dimension
        token_masks = attention_mask.bool()
        image_caption_token_masks = image_caption_att_mask.bool()
        aspect_spans, sentiment = labels[:, :, :2], labels[:, :, 2]
        caption_feats = self.caption_backbone(image_caption_token_ids, image_caption_att_mask)
        _, cls_token, h_token = self.lstm_backbone(input_ids, attention_mask)
        h_img_feats = self.vision_backbone(images)
        # resnet output reshape
        h_img_feats = h_img_feats.view(-1, 2048, 49).permute(0, 2, 1) #cmmt
        if not self.training:
            output = self.ddim_sample(h_token, token_masks, h_img_feats, caption_feats, image_caption_token_masks)
        else:
            targets = self.prepare_targets(aspect_spans, token_masks)
            output = self.head(
                targets.diffused_spans,
                h_token,
                token_masks,
                targets.timestamps,
                h_img_feats,
                caption_feats, 
                image_caption_token_masks,
            )

            pre_x_start = self.x_start_from_span(output.spans, token_masks)
            pre_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pre_x_start, x_t=targets.x_t, t=targets.timestamps
            )

            loss = self.loss_fn(
                DiffusionLMLossInput(
                    output,
                    sentiment,
                    aspect_spans,
                    token_masks,
                    targets.posterior_log_variance_clipped,
                    targets.posterior_mean,
                    pre_mean,
                    pre_x_start,
                    targets.timestamps,
                )
            )
            output = output._replace(loss=loss)

        if "word_ids" in kwargs:
            output = output._replace(word_ids=kwargs["word_ids"])

        return output

