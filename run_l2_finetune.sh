#!/bin/bash
#SBATCH --job-name=l2_finetune
#SBATCH --output=logs/l2_finetune_%j.out
#SBATCH --error=logs/l2_finetune_%j.err

# UPDATE THIS PATH after L1 training completes
L1_CHECKPOINT="/home/s52melba/CerraData_Project_Phenorob/checkpoints_data_splitted/l1_pretrain_7classes_splitted_20251126_144207/last.ckpt"

echo "=== L2 FINE-TUNING EXPERIMENT (FROM L1 PRETRAINING) ==="
echo "Starting at: $(date)"
echo "Using L1 checkpoint: $L1_CHECKPOINT"

python train_l2_finetune.py \
    --l1_checkpoint "$L1_CHECKPOINT" \
    --data_dir /home/s52melba/CerraData_Project_Phenorob/cerradata_splitted \
    --experiment_name "l2_finetune_14classes_from_l1_datasplitted_5percent" \
    --gpu_ids "0" \
    --batch_size 100 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --checkpoint_dir ./checkpoints_data_splitted \
    --log_dir ./logs_splitted \
    --data_percentage 5

echo "Completed at: $(date)"