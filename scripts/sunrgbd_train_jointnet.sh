#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --batchSize 1 --testBatchSize 1 --threads 8 --nEpochs 15 --dataset sunrgbd \
    --branch jointnet --rate_decay 2 --fine_tune True --lr 0.0001 \
    --pre_train_model_path suncg/models_final/posenet_5_8.pth --pre_train_model_path_2 suncg/models_final/bdbnet_5_8.pth
