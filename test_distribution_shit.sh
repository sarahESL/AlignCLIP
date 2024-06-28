#!/bin/bash

OUTPATH=path/to/outputs/downstreams/cls_zeroshoshot
BS=128
MODEL="ViT-B-16"
#MODEL="ViT-B-16-512"
#MODEL="ViT-L-16"

CHKPNT=path/to/trained/checkpoints/epoch_30.pt

SKETCH=path/to/datasets/imagenet_sketch/imagenet-sketch/sketch
V2=path/to/datasets/imagenetv2-matched-frequency-format-val
A=path/to/datasets/imagenet-a
O=path/to/datasets/imagenet-o
R=path/to/datasets/imagenet_r/imagenet-r

python -m main.run --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-sketch=$SKETCH
python -m main.run --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-a=$A
python -m main.run --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-o=$O
python -m main.run --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-r=$R
python -m main.run --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --imagenet-v2=$V2
