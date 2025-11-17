#!/bin/bash
#SBATCH --job-name=l1_pretrain
#SBATCH --output=logs/l1_pretrain_%j.out
#SBATCH --error=logs/l1_pretrain_%j.err

echo "=== L1 PRETRAINING EXPERIMENT ==="
echo "Starting at: $(date)"

python train_baseline.py \
    --data_dir /home/s52melba/CerraData_Project_Phenorob/kaggle_temp/cerradata_4mm \
    --label_level L1 \
    --experiment_name "l1_pretrain_7classes" \
    --gpu_ids "0" \
    --batch_size 512 \
    --num_epochs 100 \
    --learning_rate 1e-3 \
    --checkpoint_dir ./checkpoints \
    --log_dir ./logs

echo "Completed at: $(date)"