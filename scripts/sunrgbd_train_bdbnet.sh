#/bin/bash
CUDA_VISIBLE_DEVICES=0 python train.py --batchSize 2 --testBatchSize 2 --threads 8 --nEpochs 100 --dataset sunrgbd --branch bdbnet --rate_decay 10 --fine_tune False --lr 0.001
