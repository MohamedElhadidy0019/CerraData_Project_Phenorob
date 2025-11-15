#!/bin/bash
#SBATCH --job-name=l2_finetune
#SBATCH --output=logs/l2_finetune_%j.out
#SBATCH --error=logs/l2_finetune_%j.err

# UPDATE THIS PATH after L1 training completes
L1_CHECKPOINT="./checkpoints/baseline_l1_7class_TIMESTAMP/best.ckpt"

echo "=== L2 FINE-TUNING EXPERIMENT (FROM L1 PRETRAINING) ==="
echo "Starting at: $(date)"
echo "Using L1 checkpoint: $L1_CHECKPOINT"

python train_l2_finetune.py \
    --l1_checkpoint "$L1_CHECKPOINT" \
    --data_dir /home/s52melba/CerraData_Project_Phenorob/kaggle_temp/cerradata_4mm \
    --experiment_name "l2_finetune_14classes_from_l1" \
    --gpu_ids "1" \
    --batch_size 256 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --checkpoint_dir ./checkpoints \
    --log_dir ./logs

echo "Completed at: $(date)"