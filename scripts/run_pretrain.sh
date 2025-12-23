#! /bin/bash

export CUDA_VISIBLE_DEVICES=
export WANDB_API_KEY=
export WANDB_MODE=online

cd /your/path/to/CVPR2022-Pretrained-ViT-PyTorch # change to project directory
source /your/path/to/conda/bin/activate vit_env  # activate conda environment

# Pre-training dataset MVFractalDB/rcdb 1000/21k
PRETRAIN=MVFractalDB
CLASSES=1000
# model size
MODEL=deit_base_patch16_224
# initial learning rate
LR=1.0e-3
# num of epochs
EPOCHS=90
# path to train dataset
SOURCE_DATASET=./dataset/${PRETRAIN}-${CLASSES}/images
# output dir path
OUT_DIR=./check_points/${MODEL}/${CLASSES}/pretrain
# num of GPUs
NGPUS=2
# num of processes per node
NPERNODE=2
# local mini-batch size (global mini-batch size = NGPUS × LOCAL_BS)
LOCAL_BS=64

# environment variable which is the IP address of the machine in rank 0 (need only for multiple nodes)
# MASTER_ADDR="192.168.1.1"
# or export MASTER_PORT=29500

mpirun -npernode ${NPERNODE} -np ${NGPUS} \
python pretrain.py ${SOURCE_DATASET} \
    --model ${MODEL} --experiment pretrain_${MODEL}_${PRETRAIN}${CLASSES}_${LR} \
    --input-size 3 224 224 \
    --sched cosine_iter --epochs ${EPOCHS} --lr ${LR} --weight-decay 0.05 \
    --batch-size ${LOCAL_BS} --opt adamw --num-classes ${CLASSES} \
    --warmup-epochs 5 --cooldown-epochs 0 \
    --smoothing 0.1 --drop-path 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --remode pixel --interpolation bicubic --hflip 0.0 \
    -j 8 --eval-metric loss \
    --interval-saved-epochs 10 --output ${OUT_DIR} \
    --log-wandb