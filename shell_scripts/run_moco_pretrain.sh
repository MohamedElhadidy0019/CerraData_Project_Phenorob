#!/bin/bash

echo "=== MOCO V2 SELF-SUPERVISED PRETRAINING ==="
echo "Starting at: $(date)"

# Stronger augmentations + larger memory bank + lower temperature
# Following recommendations to address dimensional collapse
python train_moco_lightly.py \
    --data_dir /home/s52melba/CerraData_Project_Phenorob/cerradata_splitted \
    --experiment_name "moco_pretrain_resnet34_v2_more_aggressive" \
    --gpu_ids "0" \
    --batch_size 256 \
    --num_epochs 1000 \
    --learning_rate 0.05 \
    --temperature 0.07 \
    --projection_dim 128 \
    --memory_bank_size 65536 \
    --num_workers 32 \
    --prefetch_factor 2 \
    --checkpoint_dir ./checkpoints_data_splitted \
    --log_dir ./logs_splitted

echo "Completed at: $(date)"
