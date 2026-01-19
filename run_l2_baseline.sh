#!/bin/bash
#SBATCH --job-name=l2_baseline
#SBATCH --output=logs/l2_baseline_%j.out
#SBATCH --error=logs/l2_baseline_%j.err

echo "=== L2 BASELINE EXPERIMENT (NO PRETRAINING) ==="
echo "Starting at: $(date)"

python train_baseline.py \
    --data_dir /home/s52melba/CerraData_Project_Phenorob/cerradata_splitted \
    --label_level L2 \
    --experiment_name "l2_baseline_14classes_no_pretrain_classes_splitted_5percenet" \
    --gpu_ids "0" \
    --batch_size 100 \
    --num_epochs 200 \
    --learning_rate 1e-3 \
    --checkpoint_dir ./checkpoints_data_splitted \
    --log_dir ./logs_splitted \
    --data_percentage 5

echo "Completed at: $(date)"