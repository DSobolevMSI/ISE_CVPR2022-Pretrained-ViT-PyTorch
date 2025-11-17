#! /bin/bash
# number of classes
CLASSES=32
# model size
MODEL=base
# path to test dataset
SOURCE_DATASET=./dataset/SO32/val

## still working on inference output format
python validate.py \
    --data-dir ${SOURCE_DATASET} \
    --model deit_${MODEL}_patch16_224 \
    --num-classes ${CLASSES} \
    --class-map ./dataset/SO32_class_map.txt \
    --checkpoint ./check_points/base/32/finetune/finetune_rcdb_deit_base_SO32_1.0e-3/model_best.pth.tar