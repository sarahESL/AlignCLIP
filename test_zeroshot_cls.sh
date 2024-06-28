#!/bin/bash

OUTPATH=path/to/outputs/downstreams/cls_zeroshoshot
BS=4096
MODEL="ViT-B-16"
#MODEL="ViT-B-16-512"
#MODEL="ViT-L-16"

CHKPNT=path/to/trained/checkpoints/epoch_30.pt

CIFAR10=path/to/datasets/cifar10
CIFAR100=path/to/datasets/cifar100
IMAGENETVAL=path/to/datasets/imagenet_1k/val
IMAGENETTRAIN=path/to/datasets/imagenet_1k/train
FLOWERS102=path/to/datasets/flower_102/dataset
FOOD101=path/to/datasets/food_101
STANFORD=path/to/datasets/stanford


python -m main.run --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --cifar10=$CIFAR10
python -m main.run --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --cifar100=$CIFAR100
python -m main.run --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-val=$IMAGENETVAL --imagenet-train=$IMAGENETTRAIN
python -m main.run --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --flowers-102=$FLOWERS102
python -m main.run --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --food-101=$FOOD101 
python -m main.run --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --stanford=$STANFORD 
