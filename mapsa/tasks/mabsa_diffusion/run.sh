#!/bin/bash
# TODO search by hydra
script_dir=$(dirname "$0")
# Only Text Model
# model_names=("roberta-base" "bert-base-uncased" "albert-base-v1" "albert-base-v2" "bert-large-uncased" "xlm-roberta-base" "xlm-roberta-large-finetuned-conll03-english")

# Text-Image Model
# model_names=( "microsoft/resnet-152" "google/vit-base-patch16-224" "openai/clip-vit-base-patch32" "openai/clip-vit-large-patch14" "Salesforce/blip-itm-base-coco" "Salesforce/blip-itm-large-coco" "Salesforce/blip-itm-large-flickr")
model_names=("roberta-base")
image_model_names=("microsoft/resnet-152")
# datasets=("twitter15" "twitter17")
# datasets=("twitter15")
datasets=("twitter17")
labels=("SENTIMENT")
sub_modules=("NONE")
repeat_gt_nums=(80)
export HF_HOME=/home/haomei/raid1/vscodeproject/MAPSA/mapsa_exp
export HF_ENDPONT=https://hf-mirror.com/
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=true
# export CUDA_VISIBLE_DEVICES=0
# export TRANSFORMERS_VERBOSITY=detail

for model_name in "${model_names[@]}"
do
    for image_model_name in "${image_model_names[@]}"
    do
        for label in "${labels[@]}"
        do
            for module in "${sub_modules[@]}"
            do
                for dataset in "${datasets[@]}"
                do
                    for gt_nums in "${repeat_gt_nums[@]}"
                    do
                        echo "Train model: $model_name, Image model: $image_model_name Label type: $label, Sub module: $module"
                        python "$script_dir"/main.py --lm-name "$model_name" --im-name "$image_model_name" --label-type "$label" --module-type "$module" --dataset "/home/haomei/raid1/vscodeproject/dataset/twitter-dataset/$dataset/npz-caption" --repeat-gt-nums $gt_nums --re-gen-input-image --skip-train
                    done
                done
            done
        done
    done
done
