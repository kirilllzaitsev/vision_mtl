#!/bin/bash

script_dir=$(dirname $0)
vision_mtl_dir=$(dirname $script_dir)/vision_mtl

cd "$vision_mtl_dir" || exit 1

exp_tags=
backbone_weights=imagenet
model_name=basic
num_epochs=50
batch_size=4

python training_lit.py --do_plot_preds --exp_tags "$exp_tags" --num_workers 2 --batch_size "$batch_size"  --val_epoch_freq 1 --save_epoch_freq 5 --num_epochs "$num_epochs" --lr 5e-4 --model_name "$model_name" --backbone_weights "$backbone_weights" --dataset_name nyuv2 --device cuda:0
