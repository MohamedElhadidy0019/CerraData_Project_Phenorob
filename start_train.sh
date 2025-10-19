#!/bin/bash
python train_baseline.py \
    --data_dir /scratch/s52melba/CerraData_Project_Phenorob/kaggle_temp/cerradata_4mm \
    --gpu_ids 1 \
    --batch_size 256 \
    --num_epochs 1000 \
    --dropout_rate 0.3 \
    --weight_decay 1e-3 \
    --experiment_name "unet_resnet34_dropout_wd1e3"[00