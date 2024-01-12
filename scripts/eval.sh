#!/bin/bash
echo "Evaluating..."
config="configs/faster_rcnn_R50_BDD.yaml"
weights="./output/faster_rcnn_R50_BDD/model_best.pth"
OUTPUT_DIR="./output"

python train.py \
       --config ${config} \
       --eval-only \
       MODEL.WEIGHTS ${weights} \
       OUTPUT_DIR ${OUTPUT_DIR}
