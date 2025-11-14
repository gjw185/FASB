#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
model_name="llama2_chat_7B"
num_heads=(24)
device=0
thresholds=(0.6)
num_fold=2
use_center_of_mass="--use_center_of_mass"
alpha_values=(60)
back_nums=(10)
used_ratio=(1.0)

for alpha in "${alpha_values[@]}"; do
    for back_num in "${back_nums[@]}"; do
        for threshold in "${thresholds[@]}"; do
            python FASB.py $model_name \
            --num_heads $num_heads \
            --back_num $back_num \
            --alpha $alpha \
            --threshold $threshold \
            --device $device \
            --num_fold $num_fold \
            $use_center_of_mass
        done
    done
done