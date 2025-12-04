#!/bin/bash
#SBATCH --job-name=l1_pretrain
#SBATCH --output=logs/l1_pretrain_%j.out
#SBATCH --error=logs/l1_pretrain_%j.err

echo "=== L1 PRETRAINING EXPERIMENT ==="
echo "Starting at: $(date)"

python train_baseline.py \
    --data_dir /home/s52melba/CerraData_Project_Phenorob/cerradata_splitted \
    --label_level L1 \
    --experiment_name "l1_pretrain_7classes_splitted" \
    --gpu_ids "0" \
    --batch_size 100 \
    --num_epochs 100 \
    --learning_rate 1e-3 \
    --checkpoint_dir ./checkpoints_data_splitted \
    --log_dir ./logs_splitted

echo "Completed at: $(date)"