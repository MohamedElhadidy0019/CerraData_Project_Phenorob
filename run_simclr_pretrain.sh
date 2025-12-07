#!/bin/bash

echo "=== SIMCLR SELF-SUPERVISED PRETRAINING ==="
echo "Starting at: $(date)"

python train_simclr_lightly.py \
    --data_dir /home/s52melba/CerraData_Project_Phenorob/cerradata_splitted \
    --experiment_name "simclr_pretrain_resnet34" \
    --gpu_ids "0" \
    --batch_size 100 \
    --num_epochs 200 \
    --learning_rate 1e-3 \
    --temperature 0.5 \
    --projection_dim 128 \
    --num_workers 4 \
    --checkpoint_dir ./checkpoints_data_splitted \
    --log_dir ./logs_splitted

echo "Completed at: $(date)"
