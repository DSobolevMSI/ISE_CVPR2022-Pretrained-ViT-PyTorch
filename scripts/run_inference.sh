#! /bin/bash

export WANDB_API_KEY=
export WANDB_MODE=online
export WANDB_PROJECT=

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
    --class-map ./dataset/SO32_class_map.txt \
    --checkpoint ./check_points/base/32/finetune/finetune_deit_base_SO3232_1.0e-3/model_best.pth.tar  \
    --results-dir ./inference_results/${DATA_NAME}${CLASSES}    \
    --fullname --include-index