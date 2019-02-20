#/bin/bash
CUDA_VISIBLE_DEVICES=0 python train.py --batchSize 32 --testBatchSize 32 --threads 8 --nEpochs 200 --dataset sunrgbd --branch posenet --rate_decay 10 --fine_tune True --pre_train_model_path suncg/models_final/posenet_5_8.pth --lr 0.0001
