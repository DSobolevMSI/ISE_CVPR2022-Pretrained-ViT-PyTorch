#! /bin/bash

export WANDB_API_KEY=
export WANDB_MODE=online
export WANDB_PROJECT=

export MASTER_PORT=29500

CLASSES=32
SAVE_ROOT=./dataset/SO32

# MV-FractalDB Pre-training
# model size
MODEL=base
# initial learning rate
LR=1.0e-3
# name of dataset
DATA_NAME=SO
# num of epochs
EPOCHS=300
# path to train dataset
SOURCE_DATASET=${SAVE_ROOT}
# output dir path
OUT_DIR=./check_points/${MODEL}/${CLASSES}/finetune
# num of GPUs
NGPUS=2
# num of processes per node
NPERNODE=2
# local mini-batch size (global mini-batch size = NGPUS × LOCAL_BS)
LOCAL_BS=64

# environment variable which is the IP address of the machine in rank 0 (need only for multiple nodes)
# MASTER_ADDR="192.168.1.1"

mpirun -npernode ${NPERNODE} -np ${NGPUS} \
python finetune.py ${SOURCE_DATASET} \
    --model deit_${MODEL}_patch16_224 --experiment finetune_deit_${MODEL}_${DATA_NAME}${CLASSES}_${LR} \
    --input-size 3 224 224 \
    --sched cosine_iter --epochs ${EPOCHS} --lr ${LR} --weight-decay 0.05 \
    --batch-size ${LOCAL_BS} --opt adamw --num-classes ${CLASSES} \
    --warmup-epochs 5 --cooldown-epochs 0 \
    --smoothing 0.1 --drop-path 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --remode pixel --interpolation bicubic --hflip 0.0 \
    -j 8 --eval-metric loss \
    --interval-saved-epochs 50 --output ${OUT_DIR} \
    --log-wandb \
    --pretrained-path ./ckpts/pretrained_exfractal_21k_base.pth.tar
