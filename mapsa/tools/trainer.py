from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import Seq2SeqTrainer as TFSeq2SeqTrainer
from transformers import Trainer


class Seq2SeqWithVisionTrainer(TFSeq2SeqTrainer):

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs
        )
        #     with open('img_id.txt', 'a+') as fp:
        #         fp.write('\n'.join(inputs['img_id'] + ['\n']))
        return (
            loss,
            generated_tokens,
            {
                "labels": labels,
                "img_label": inputs["img_label"],
                #         'img_id': inputs['img_id']
            },
        )


class DiffusionWithVisionTrainer(Trainer):

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        from transformers.trainer import nested_numpify
        import numpy as np
        from mapsa.tools.eval import remove_overlapping_spans, fix_spans_by_word_ids

        loss, logits, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs
        )

        boundary_threshold = 0
        sentiment_threshold = 1.8
        ignore_id = -100

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
            ) & ((l_score + r_score + cls_score) > sentiment_threshold
             )

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
        pred_span_words = []
        label_span_words = []
        for i, (pred, label) in enumerate(
            zip(zip(*nested_numpify(logits)), nested_numpify(labels))
        ):
            word_ids, *_ = pred
            pred_spans.append(_format_pred(pred))
            label_spans.append(_format_label(label))
            pred_span_words.append(
                set(
                    [
                        " ".join(
                            inputs["words"][i][word_ids[se[0]] : word_ids[se[1]] + 1]
                        )
                        for se in pred_spans[-1]
                        if se
                    ]
                )
            )
            label_span_words.append(
                set(
                    [
                        " ".join(
                            inputs["words"][i][word_ids[se[0]] : word_ids[se[1]] + 1]
                        )
                        for se in label_spans[-1]
                        if se
                    ]
                )
            )

        with open("img_id_and_spans.txt", "a+", encoding="utf-8") as fp:
            for line in zip(
                inputs["image_ids"],
                inputs["words"],
                pred_spans,
                label_spans,
                pred_span_words,
                label_span_words,
            ):
                # fp.write("|".join(line) + "\n")
                # fp.write("|".join(str(item) if isinstance(item, set) else item for item in line) + "\n")

                fp.write(
                    "\n".join(
                        (
                            str(item.tolist())
                            if isinstance(item, np.ndarray)
                            else str(item) if isinstance(item, set) else str(item)
                        )  # 确保所有其他类型也被转换为字符串
                        for item in line
                    )
                    + "\n\n"
                )

        return loss, logits, labels
