#!/bin/bash
#SBATCH --job-name=l2_finetune_loop
#SBATCH --output=logs/l2_finetune_loop_%j.out
#SBATCH --error=logs/l2_finetune_loop_%j.err

# UPDATE THIS PATH to your best L1 checkpoint
L1_CHECKPOINT="/home/s52melba/CerraData_Project_Phenorob/CerraData-4MM/experiment_results/weights/l1_pretrain_7classes_multimodal_20260130_225343/last.ckpt"

echo "=== L2 FINE-TUNING EXPERIMENT (FROZEN ENCODER) - MULTIPLE DATA PERCENTAGES ==="
echo "Starting at: $(date)"
echo "Using L1 checkpoint: $L1_CHECKPOINT"

# Define percentages to test
PERCENTAGES="0.5 1 2.5 3.5 5 10 25 50"

# Early stopping patience
PATIENCE=50

# Organized directories for scaling experiments
LOG_BASE="/home/s52melba/CerraData_Project_Phenorob/CerraData-4MM/experiment_diff_percentages/logs/finetune_frozen"
CHECKPOINT_BASE="/home/s52melba/CerraData_Project_Phenorob/CerraData-4MM/experiment_diff_percentages/weights/finetune_frozen"

# Create directories if they don't exist
mkdir -p "$LOG_BASE"
mkdir -p "$CHECKPOINT_BASE"

echo "Logs will be saved to: $LOG_BASE"
echo "Checkpoints will be saved to: $CHECKPOINT_BASE"

for PCT in $PERCENTAGES; do
    # Convert percentage to safe filename (replace . with p)
    PCT_NAME=$(echo $PCT | sed 's/\./_/g')

    echo ""
    echo "========================================="
    echo "Running L2 Fine-tuning with ${PCT}% data"
    echo "========================================="

    python train_l2_finetune.py \
        --l1_checkpoint "$L1_CHECKPOINT" \
        --data_dir /home/s52melba/CerraData_Project_Phenorob/CerraData-4MM/dataset_splitted \
        --experiment_name "l2_finetune_14classes_frozenenc_${PCT_NAME}percent" \
        --gpu_ids "0" \
        --batch_size 100 \
        --num_epochs 500 \
        --learning_rate 1e-4 \
        --checkpoint_dir "$CHECKPOINT_BASE" \
        --log_dir "$LOG_BASE" \
        --data_percentage $PCT \
        --patience $PATIENCE \
        --seed 42 \
        --freeze_encoder

    echo "Completed ${PCT}% at: $(date)"
done

echo ""
echo "=== ALL L2 FINE-TUNING EXPERIMENTS COMPLETED ==="
echo "Finished at: $(date)"
