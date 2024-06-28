#!/bin/bash


OUTPATH=path/to/output/logs
BS=512
LR=1e-3
N_EPOCHS=30
MODEL="ViT-B-16"
#MODEL="ViT-B-16-512"
#MODEL="ViT-L-16"
TRAIN_DATA="path/to/cc12m/{00000..01242}.tar"
PROJECT_NAME=sharedCLIP

wandb login $(cat ~/.wandb_secret)

python -m main.run --logs=$OUTPATH --save-frequency 2 --report-to wandb --wandb-project-name=$PROJECT_NAME --train-data=$TRAIN_DATA --train-num-samples 10030127 --warmup 10000  --batch-size=$BS --lr=$LR --wd=0.1 --epochs=$N_EPOCHS --workers=2 --model $MODEL --precision amp --dataset-type webdataset
