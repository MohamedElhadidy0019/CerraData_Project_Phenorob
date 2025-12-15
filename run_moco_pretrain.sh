#!/bin/bash

echo "=== MOCO V2 SELF-SUPERVISED PRETRAINING ==="
echo "Starting at: $(date)"

python train_moco_lightly.py \
    --data_dir /home/s52melba/CerraData_Project_Phenorob/cerradata_splitted \
    --experiment_name "moco_pretrain_resnet34" \
    --gpu_ids "0" \
    --batch_size 256 \
    --num_epochs 1000 \
    --learning_rate 0.03 \
    --temperature 0.2 \
    --projection_dim 128 \
    --memory_bank_size 8192 \
    --num_workers 32 \
    --prefetch_factor 2 \
    --checkpoint_dir ./checkpoints_data_splitted \
    --log_dir ./logs_splitted

echo "Completed at: $(date)"
