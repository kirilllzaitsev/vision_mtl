#!/bin/bash

script_dir=$(dirname $0)
vision_mtl_dir=$(dirname $script_dir)/vision_mtl

cd "$vision_mtl_dir" || exit 1

exp_tags=
backbone_weights=imagenet
model_name=mtan
num_epochs=20
batch_size=8
num_workers=4

python training_lit.py --do_plot_preds --exp_tags "$exp_tags" --num_workers "$num_workers" --batch_size "$batch_size" --val_epoch_freq 1 --save_epoch_freq 5 --num_epochs "$num_epochs" --lr 5e-4 --model_name "$model_name" --backbone_weights "$backbone_weights" --dataset_name cityscapes --device cuda:0
