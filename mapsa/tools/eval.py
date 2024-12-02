from typing import List

import evaluate
import numpy as np
import torch
from transformers import EvalPrediction

from mapsa.data.data_types import MetricType
from mapsa.tools.tokenizer import ToknizerWrapper


def compute_classification_metric(
    p: EvalPrediction, tokenizer: ToknizerWrapper = None, ignore_id=-100
):
    predictions, labels = p
    if len(predictions.shape) == 3:
        predictions = np.argmax(predictions, axis=-1)
    # Remove ignored index (special tokens)
    true_predictions = [
        [
            tokenizer.reverse_labels_mapping[p]
            for (p, l) in zip(prediction, label)
            if l != ignore_id
        ]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [tokenizer.reverse_labels_mapping[l] for l in label if l != ignore_id]
        for label in labels
    ]
    metric = evaluate.load("seqeval")
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
        "acc_and_f1": (results["overall_f1"] + results["overall_accuracy"]) / 2,
    }


def compute_span_metric_core(predictions, labels, coarse_st_index=None):
    correct_pred, total_gt, total_pred = 0.0, 0.0, 0.0
    correct_acc = 0.0  # 新增准确率相关变量
    total_samples = len(predictions)  # 总样本数，用于计算准确率
    for pred_spans, label_spans in zip(predictions, labels):
        # MABSA
        unique_pred_spans = set(map(tuple, filter(None, pred_spans)))
        unique_label_spans = set(map(tuple, filter(None, label_spans)))

        # MATE
        # 只取前两个标签 (l, r)，忽略情感标签 (s)
        # unique_pred_spans = set(tuple(span[:2]) for span in pred_spans if span)  # 保留前两项
        # unique_label_spans = set(tuple(span[:2]) for span in label_spans if span)  # 保留前两项

        total_pred += len(unique_pred_spans)
        total_gt += len(unique_label_spans)

        # 计算准确预测的样本数
        if unique_pred_spans == unique_label_spans:
            correct_acc += 1

        if coarse_st_index is None:
            correct_pred += len(unique_pred_spans & unique_label_spans)
        else:
            unique_label_spans = list(unique_label_spans)
            index_labels = [lb[:coarse_st_index] for lb in unique_label_spans]
            for pred in unique_pred_spans:
                if pred[:coarse_st_index] not in index_labels:
                    continue

                i_lb = index_labels.index(pred[:coarse_st_index])
                flag = 1
                for coarse_pred, coarse_label in zip(
                    pred[coarse_st_index:],
                    unique_label_spans[i_lb][coarse_st_index:],
                ):
                    if len(set(coarse_pred) & set(coarse_label)) == 0:
                        flag = 0
                        break

                correct_pred += flag

    if correct_pred == 0:
        return {
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "accuracy": correct_acc / total_samples if total_samples > 0 else 0,  # 新增准确率
        }

    p = correct_pred / total_pred
    r = correct_pred / total_gt
    f1 = 2 * p * r / (p + r)
    acc = correct_acc / total_samples if total_samples > 0 else 0  # 准确率计算
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "accuracy": acc,  # 返回准确率
    }

def compute_sentiment_accuracy(predictions, labels):
    aspect_correct_sentiment_correct = 0.0  # aspect 正确且情感正确的数量
    aspect_correct_total = 0.0  # aspect 正确的总数量

    for pred_spans, label_spans in zip(predictions, labels):
        # 将预测和标签都转换为仅保留 (l, r) 和 (s) 的集合
        pred_aspects = set((span[0], span[1]) for span in pred_spans if span)
        label_aspects = set((span[0], span[1]) for span in label_spans if span)

        # 计算 aspect 正确的部分
        correct_aspects = pred_aspects & label_aspects  # aspect 预测正确的部分

        # 如果 aspect 正确，则计算情感标签
        for correct_aspect in correct_aspects:
            # 从原始预测和标签中找到对应的情感 (s)
            pred_sentiment = [span[2] for span in pred_spans if (span[0], span[1]) == correct_aspect]
            label_sentiment = [span[2] for span in label_spans if (span[0], span[1]) == correct_aspect]

            if pred_sentiment and label_sentiment:
                aspect_correct_total += 1
                if pred_sentiment[0] == label_sentiment[0]:
                    aspect_correct_sentiment_correct += 1

    # 计算情感准确率
    sentiment_acc = aspect_correct_sentiment_correct / aspect_correct_total if aspect_correct_total > 0 else 0
    return {
        "sentiment_accuracy": sentiment_acc  # 返回情感准确率
    }



def compute_seq_token_metric(
    p: EvalPrediction, tokenizer: ToknizerWrapper = None, ignore_id=-100
):

    def _format(input_ids):
        return [
            list(
                map(
                    str.strip,
                    filter(
                        # lambda x: len(x) > 6 and
                        # (x.count(tokenizer.pos_token) == 1 or x.count(
                        #     tokenizer.neu_token) == 1 or x.count(tokenizer.neg_token
                        #                                         ) == 1),
                        lambda x: len(x) > 6,
                        tokenizer.decode(tok_ids, skip_special_tokens=False)
                        .replace(tokenizer.bos_token, "")
                        .replace(tokenizer.eos_token, "")
                        .replace(tokenizer.pad_token, "")
                        .split(tokenizer.ssep_token),
                    ),
                )
            )
            for tok_ids in input_ids
        ]

    predictions, labels = p
    if len(predictions.shape) == 3:
        predictions = np.argmax(predictions, axis=-1)
    labels[labels == ignore_id] = tokenizer.ssep_token_id
    pred_words = _format(predictions)
    label_words = _format(labels)
    return compute_span_metric_core(pred_words, label_words)


"""
pred_left: torch.Tensor = None
pred_right: torch.Tensor = None
pred_logits: torch.Tensor = None
pred_spans: torch.Tensor = None
pred_x_start: torch.Tensor = None
pred_noise: torch.Tensor = None
"""


def compute_span_metric(
    p: EvalPrediction, tokenizer: ToknizerWrapper = None, ignore_id=-100
):
    predictions, labels = p
    step = 3

    def _format_pred(pred):
        spans = []
        # next of bos
        for i in range(1, len(pred), step):
            if (
                pred[i] >= len(pred)
                or pred[i] == tokenizer.pad_token_id
                or pred[i] == tokenizer.eos_token_id
            ):
                continue
            span = pred[i : i + step].tolist()
            if len(span) < 3:
                continue
            spans.append(span)
        return set(map(tuple, spans))

    def _format_label(label):
        spans = []
        for i in range(0, len(label), step):
            if (
                label[i] >= len(label)
                or label[i] == ignore_id
                or label[i] == tokenizer.pad_token_id
                or label[i] == tokenizer.eos_token_id
            ):
                continue
            span = label[i : i + step].tolist()
            if len(span) < 3:
                continue
            spans.append(span)
        return set(map(tuple, spans))

    pred_spans = []
    label_spans = []
    for prediction, label in zip(predictions, labels):
        pred_spans.append(_format_pred(prediction))
        label_spans.append(_format_label(label))

    return compute_span_metric_core(pred_spans, label_spans)


def compute_two_stage_metric(
    p: EvalPrediction, tokenizer: ToknizerWrapper = None, ignore_id=-100
):
    predictions, labels = p
    pred_spans = []
    label_spans = []
    for prediction, label in zip(predictions, labels):
        pred_spans.append(
            [
                (int(st), int(ed), int(polarity))
                for st, ed, polarity in prediction
                if st != ignore_id and ed != ignore_id
            ]
        )
        label_spans.append(
            [
                (int(st), int(ed), int(polarity))
                for st, ed, polarity in label
                if st != ignore_id and ed != ignore_id
            ]
        )
    return compute_span_metric_core(pred_spans, label_spans)


def compute_span_text_img_metric(
    p: EvalPrediction, tokenizer: ToknizerWrapper = None, ignore_id=-100
):

    def _token_id_to_label(input_ids, img_label: torch.Tensor = None):

        def _decode(tok_ids: np.ndarray):
            vsep_index = -1
            if tokenizer.vsep_token_id in tok_ids:
                vsep_index = tok_ids.tolist().index(tokenizer.vsep_token_id)
            sentence = (
                tokenizer.decode(
                    tok_ids[:vsep_index],
                    skip_special_tokens=False,
                )
                .replace(
                    tokenizer.bos_token,
                    "",
                )
                .replace(
                    tokenizer.eos_token,
                    "",
                )
                .replace(
                    tokenizer.pad_token,
                    "",
                )
            )
            # ... | tokenizer.ssep_token | ...
            res_items = list(
                map(
                    lambda x: tokenizer.label_convertor.decode(x).tolist(),
                    sentence.split(tokenizer.ssep_token),
                )
            )
            if vsep_index != -1:
                img_region_part = tok_ids[vsep_index + 1 :]
                vis_idx = 0
                for item in res_items:
                    # in image
                    if not item or len(item) < 3:
                        continue
                    if (
                        item[2]
                        and len(img_region_part) > vis_idx
                        and img_region_part[vis_idx] != tokenizer.vsep_token_id
                    ):
                        item.append((img_region_part[vis_idx],))
                        vis_idx += 1
                    else:
                        item.append((None,))
            print("res_items:", res_items)
            return res_items

        formated_res = [_decode(tok_ids) for tok_ids in input_ids]

        # Prepare label
        if img_label is not None:
            for fr, ilb in zip(formated_res, img_label):
                img_region_mask = ilb.sum(-1) != 0
                img_region_part = ilb[img_region_mask]
                vis_idx = 0
                for item in fr:
                    if item[2] and len(img_region_part) > vis_idx:
                        # item.append((img_region_part[vis_idx].argmax(),))
                        item.append(tuple(np.where(img_region_part[vis_idx] > 0)[0]))
                        vis_idx += 1
                    else:
                        item.append((None,))
        # TODO
        return [
            [rr for rr in r if rr[2] != None] for r in formated_res
        ]  # entity-object-sa eval
        # return [[tuple(rr[:2]) for rr in r] for r in formated_res] # sa eval
        # return [[tuple(rr[:2]) for rr in r if rr[2] != None] for r in formated_res]  # sa eval for region is not none

    predictions, labels = p
    print("predictions:", predictions)
    print("labels:", labels)
    text_label = labels["labels"]
    text_label[text_label == ignore_id] = tokenizer.ssep_token_id
    formated_preds = _token_id_to_label(predictions)
    formated_labels = _token_id_to_label(text_label, labels["img_label"])
    return compute_span_metric_core(formated_preds, formated_labels, coarse_st_index=-1)


def remove_overlapping_spans(spans) -> np.ndarray:

    def check_partial_overlap(e1, e2):
        return (e1[0] <= e2[0] <= e1[1]) or (e2[0] <= e1[0] <= e2[1])

    non_overlapping_spans = []
    for span in spans:
        if not any(
            check_partial_overlap(span, existing) for existing in non_overlapping_spans
        ):
            non_overlapping_spans.append(span)

    return np.array(non_overlapping_spans)


def fix_spans_by_word_ids(
    spans: np.ndarray, word_ids: np.ndarray, ignore_index=-100
) -> np.ndarray:
    if isinstance(word_ids, np.ndarray):
        word_ids = word_ids.tolist()
    fixed_spans = []
    reveser_wids = word_ids[::-1]
    for span in spans:
        st, end, *elems = span
        st_wid = word_ids[int(st)]
        ed_wid = word_ids[int(end)]
        if st_wid == ignore_index or ed_wid == ignore_index:
            continue
        new_st = word_ids.index(st_wid)
        new_end = len(reveser_wids) - reveser_wids.index(ed_wid) - 1
        fixed_spans.append([new_st, new_end, *elems])
    return np.array(fixed_spans)


def compute_span_metric_from_diffusion(
    p: EvalPrediction, tokenizer: ToknizerWrapper = None, ignore_id=-100
):
    predictions, labels = p
    step = 3
    boundary_threshold = 0
    sentiment_threshold = 1.8

    def _format_pred(pred: List[np.ndarray]):
        (
            word_ids,
            l,
            r,
            logits,
            spans,
            st,
            noies,
            left_variance,
            right_variance,
            logits_variance,
        ) = pred
        l_index, r_index = l.argmax(-1), r.argmax(-1)
        l_score, r_score = l.max(-1), r.max(-1)
        # aspect de sentiment 60*4
        soft_logits = torch.from_numpy(logits).softmax(-1).numpy()
        cls_index, cls_score = soft_logits.argmax(-1), soft_logits.max(-1)

        l_r_spans = np.stack(
            [l_index, r_index, cls_index, l_score + r_score + cls_score], axis=-1
        )

        mask = (l_index <= r_index) & (
            (l_score > boundary_threshold) | (r_score > boundary_threshold)
        ) & ((l_score + r_score + cls_score) > sentiment_threshold)

        l_r_spans = l_r_spans[mask]
        if len(l_r_spans) == 0:
            return set()
        # decrease sort by score
        l_r_spans = l_r_spans[l_r_spans[:, -1].argsort()[::-1]]
        l_r_spans = fix_spans_by_word_ids(l_r_spans, word_ids, ignore_id)
        l_r_spans = remove_overlapping_spans(l_r_spans)
        # fix by word ids
        if len(l_r_spans) == 0:
            return set()
        rt_spans = l_r_spans[:, :3].astype(int)
        return set(map(tuple, rt_spans))

    def _format_label(label):
        return set(map(tuple, label))

    pred_spans = []
    label_spans = []
    for pred, label in zip(zip(*predictions), labels):
        pred_spans.append(_format_pred(pred))
        label_spans.append(_format_label(label))
    
    # 打印结果
    import pandas as pd
    df = pd.DataFrame({"pred": pred_spans, "label": label_spans})
    df.to_csv("eval_res.csv", index=False)

    # MABSA，MATE
    return compute_span_metric_core(pred_spans, label_spans)
    # MASC
    # return compute_sentiment_accuracy(pred_spans, label_spans)


def get_eval(metric_type="classification"):
    if isinstance(metric_type, str):
        metric_type = MetricType[metric_type.upper()]
    if metric_type == MetricType.CLASSIFICATION:
        return compute_classification_metric
    if metric_type == MetricType.SEQ_TOKEN:
        return compute_seq_token_metric
    if metric_type == MetricType.SPAN:
        return compute_span_metric
    if metric_type == MetricType.SPAN_TWO_STAGE:
        return compute_two_stage_metric
    if metric_type == metric_type.SPAN_TEXT_IMG:
        return compute_span_text_img_metric
    if metric_type == metric_type.ABSA_DIFFUSION:
        return compute_span_metric_from_diffusion
    raise ValueError(f"Please check your input {list(MetricType)}")
