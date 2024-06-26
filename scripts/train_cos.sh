#!/bin/bash
num_gpus=2
gpu_ratio=$(echo "$num_gpus / 2" | bc -l) # 2 is the default number of gpus per node
echo "num_gpus: $num_gpus, default_gpus number is 2, so the gpu_ratio: $gpu_ratio"

config="configs/faster_rcnn_R50_BDD.yaml"
OUTPUT_DIR="./output"

CUDA_VISIBLE_DEVICES=2,3 python train_net.py \
       --num-gpus ${num_gpus} \
       --config ${config} \
       OUTPUT_DIR ${OUTPUT_DIR}
