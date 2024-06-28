#!/bin/bash

OUTPATH=path/to/outputs/downstreams/cls_zeroshoshot
BS=16
MODEL="ViT-B-16"
#MODEL="ViT-B-16-512"
#MODEL="ViT-L-16"

CHKPNT=path/to/trained/checkpoints/epoch_30.pt

MSCOCO=path/to/datasets/ms_coco/val2017
MSCOCO_ANNOT=path/to/datasets/ms_coco/captions_val2017.json
FLICKR=path/to/datasets//Flickr30K/flickr30k_images
FLICKR_ANNOT=path/to/datasets/Flickr30K/flickr30k_test.json

python -m main.run --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --ms-coco=$MSCOCO --ms-coco-annot=$MSCOCO_ANNOT
python -m main.run --logs=$OUTPATH --pretrained $CHKPNT --batch-size=$BS --workers=2 --model $MODEL --flickr=$FLICKR --flickr-annot=$FLICKR_ANNOT
