#! /bin/bash
export CUDA_VISIBLE_DEVICES=4

# number of classes
CLASSES=32
DATA_NAME=SO
# model size
MODEL=base
# path to test dataset
SOURCE_DATASET=./dataset/SO32/val

# environment variable which is the IP address of the machine in rank 0 (need only for multiple nodes)
# MASTER_ADDR="192.168.1.1"

## still working on inference output format
python inference.py \
    --data-dir ${SOURCE_DATASET} \
    --model deit_${MODEL}_patch16_224 \
    --num-classes ${CLASSES} \
    --checkpoint ./check_points/base/32/finetune/finetune_exfractal_deit_base_SO32_1.0e-3/model_best.pth.tar  \
    --results-dir ./inference_results/exfractal_${DATA_NAME}${CLASSES}    \
    --fullname --include-index
