#!/bin/bash
num_gpus=2
gpu_ratio=$(echo "$num_gpus / 2" | bc -l) # 2 is the default number of gpus per node
echo "num_gpus: $num_gpus, default_gpus number is 2, so the gpu_ratio: $gpu_ratio"

config="configs/faster_rcnn_R50_BDD.yaml"
resume=false
OUTPUT_DIR="./output"

python train.py \
       --num-gpus ${num_gpus} \
       --config ${config} \
       --resume ${resume} \
       OUTPUT_DIR ${OUTPUT_DIR}
