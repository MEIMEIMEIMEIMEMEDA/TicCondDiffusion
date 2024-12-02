from collections import defaultdict
import logging
from typing import Dict, List

import numpy as np
import torch

from mapsa.data.data_types import TaskType
from mapsa.data.data_types import TwitterSample

logger = logging.getLogger(__name__)

SUPPORTED_DTYPES = [
    np.float64,
    np.float32,
    np.float16,
    np.complex64,
    np.complex128,
    np.int64,
    np.int32,
    np.int16,
    np.int8,
    np.uint8,
    np.bool_,
]


def _collate(
    batch: List[TwitterSample], need_fields: List[str] = None, is_stack=True
) -> Dict[str, torch.Tensor]:
    """Custom collate function to collate a list of TwitterSample instances into
    a single TwitterSample instance."""
    collated_batch = defaultdict(list)
    # Stack or concatenate the data for each field
    for i, f in enumerate(TwitterSample._fields):
        if need_fields is not None and f not in need_fields:
            continue
        msg = f"Collate field {f}"
        logger.info(msg)

        for item in batch:
            if isinstance(item[i], np.ndarray) and item[i].dtype in SUPPORTED_DTYPES:
                collated_batch[f].append(torch.from_numpy(item[i]))
            elif item[i] is not None:
                collated_batch[f].append(item[i])
        if f not in collated_batch:
            continue
        if is_stack and isinstance(collated_batch[f][0], torch.Tensor):
            collated_batch[f] = torch.stack(collated_batch[f])

    # Post check
    missing_fields = []
    for sub_field in need_fields:
        if sub_field not in collated_batch:
            missing_fields.append(sub_field)
    # if len(missing_fields) > 0:
    # logger.warning(f'{missing_fields} must in need fields {need_fields}')
    return collated_batch


def collate_for_token_cls(batch: List[TwitterSample]):
    collated_batch = _collate(
        batch,
        ["token_ids", "attention_mask", "token_labels"],
    )
    return {
        "input_ids": collated_batch["token_ids"].to(torch.long),
        "attention_mask": collated_batch["attention_mask"].to(torch.bool),
        "labels": collated_batch["token_labels"].to(torch.long),
    }


def collate_for_text_seq2seq(batch: List[TwitterSample]):
    collated_batch = _collate(
        batch,
        [
            "token_ids",
            "attention_mask",
            "decoder_token_ids",
            "decoder_attention_mask",
            "seq_token_labels",
        ],
    )
    return {
        "input_ids": collated_batch["token_ids"],
        "attention_mask": collated_batch["attention_mask"].to(torch.bool),
        "labels": collated_batch["seq_token_labels"].to(torch.long),
    }


def collate_for_mm_two_stage(batch: List[TwitterSample]):
    collated_batch = _collate(
        batch,
        [
            "image_id",
            "cropped_images",
            "region_anp",
            "image_caption",
            "words",
            "raw_target",
            "span_labels",
        ],
        False,
    )

    return {
        "image_ids": collated_batch["image_id"],
        "words": collated_batch["words"],
        "images": collated_batch["cropped_images"],
        "image_region_anps": collated_batch["region_anp"],
        "image_captions": collated_batch["image_caption"],
        "word_labels": collated_batch["raw_target"],
        "labels": torch.stack([lb for lb in collated_batch["span_labels"]]),
    }


def collate_mm_generation_data(batch: List[TwitterSample]):
    collated_batch = _collate(
        batch,
        [
            "image_id",
            "anp",
            "token_ids",
            "attention_mask",
            "box_features",
            "box_attention_mask",
            "decoder_token_ids",
            "decoder_attention_mask",
            "image_labels",
            "seq_token_labels",
        ],
        True,
    )

    return {
        "input_ids": collated_batch["token_ids"].to(torch.long),
        "attention_mask": collated_batch["attention_mask"].to(torch.bool),
        "vis_feats": collated_batch["box_features"],
        "vis_attention_mask": collated_batch["box_attention_mask"].to(torch.bool),
        "decoder_input_ids": collated_batch["decoder_token_ids"].to(torch.long),
        "decoder_attention_mask": collated_batch["decoder_attention_mask"].to(
            torch.bool
        ),
        "img_label": collated_batch["image_labels"].to(torch.float32),
        "img_anp_label": (
            collated_batch["anp"].to(torch.float32)
            if isinstance(collated_batch["anp"], torch.Tensor)
            else torch.empty(0)
        ),
        "img_id": collated_batch["image_id"],
        "labels": collated_batch["seq_token_labels"].to(torch.long),
    }


def collate_for_mabsa(batch: List[TwitterSample], repeat_gt_nums=100):
    def gen_from_span_labels(span_labels: torch.Tensor, ignore_index=-100):
        B, L = span_labels.shape
        label_max_len = L - L % 3
        # st, ed, cls label
        reshape_label = span_labels[:, :label_max_len].clone().reshape(B, -1, 3)
        label_list = []
        for i in range(B):
            cur_label = reshape_label[i]
            cur_mask = (cur_label != ignore_index).bool().sum(1) == 3
            cur_label = cur_label[cur_mask].repeat(
                (repeat_gt_nums // cur_mask.sum()) + 1, 1
            )
            cur_label = cur_label[:repeat_gt_nums, :]
            label_list.append(cur_label)
        label = torch.stack(label_list).to(span_labels.device)
        return label

    collated_batch = _collate(
        batch,
        [
            "image_id",
            "input_image",
            "token_ids",
            "attention_mask",
            "decoder_token_ids",
            "decoder_attention_mask",
            "span_labels",
            "word_ids",
            "image_caption_token_ids",
            "image_caption_att_mask",
            "words",
        ],
        True,
    )

    label = gen_from_span_labels(collated_batch["span_labels"])

    ret = {
        "image_ids": collated_batch["image_id"],
        "images": collated_batch["input_image"],
        "input_ids": collated_batch["token_ids"],
        "attention_mask": collated_batch["attention_mask"],
        "labels": label,
        "word_ids": collated_batch["word_ids"],
        "image_caption_token_ids": collated_batch["image_caption_token_ids"],
        "image_caption_att_mask": collated_batch["image_caption_att_mask"],
        "words": collated_batch["words"],
    }
    return ret


def get_collator(task_type="TEXT_TOKEN_CLS"):
    if isinstance(task_type, str):
        task_type = TaskType[task_type.upper()]
    if task_type == TaskType.TEXT_TOKEN_CLS:
        return collate_for_token_cls
    if task_type == TaskType.TEXT_SEQ2SEQ:
        return collate_for_text_seq2seq
    if task_type == TaskType.MM_TWO_STAGE:
        return collate_for_mm_two_stage
    if task_type == TaskType.MM_GEN:
        return collate_mm_generation_data
    if task_type == TaskType.MABSA_DIFFUSION:
        return collate_for_mabsa
    raise ValueError(f"Please check your input {list(TaskType)}")
