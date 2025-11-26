#!/bin/bash
#SBATCH --job-name=l2_baseline
#SBATCH --output=logs/l2_baseline_%j.out
#SBATCH --error=logs/l2_baseline_%j.err

echo "=== L2 BASELINE EXPERIMENT (NO PRETRAINING) ==="
echo "Starting at: $(date)"

python train_baseline.py \
    --data_dir ./data_split \
    --label_level L2 \
    --experiment_name "l2_baseline_14classes_no_pretrain" \
    --gpu_ids "0" \
    --batch_size 512 \
    --num_epochs 100 \
    --learning_rate 1e-3 \
    --checkpoint_dir ./checkpoints \
    --log_dir ./logs \
    --data_percentage 10

echo "Completed at: $(date)"