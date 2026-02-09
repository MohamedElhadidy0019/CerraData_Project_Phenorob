#!/bin/bash
#SBATCH --job-name=l2_from_ssl_loop
#SBATCH --output=logs/l2_from_ssl_loop_%j.out
#SBATCH --error=logs/l2_from_ssl_loop_%j.err

# UPDATE THIS PATH after MoCo pretraining completes!
# The encoder will be saved in: experiment_results/weights/moco/moco_pretrain_14ch_TIMESTAMP/encoder_final.pth
MOCO_ENCODER="/home/s52melba/CerraData_Project_Phenorob/CerraData-4MM/experiment_results/weights/moco_pretrain_14ch_multimodal_aggressive_20260208_204941/encoder_final.pth"

echo "=== L2 FINE-TUNING FROM MOCO (14-channel) - MULTIPLE DATA PERCENTAGES ==="
echo "Using encoder: $MOCO_ENCODER"
echo "Normalization: z_score"
echo "Encoder: FROZEN (only training decoder)"
echo "Starting at: $(date)"

# Define percentages to test
PERCENTAGES="0.5 1 2.5 3.5 5 10 25 50"

# Early stopping patience
PATIENCE=50

# Organized directories for scaling experiments (14-channel multimodal)
LOG_BASE="/home/s52melba/CerraData_Project_Phenorob/CerraData-4MM/experiment_diff_percentages/logs/frozen_moco_encoder"
CHECKPOINT_BASE="/home/s52melba/CerraData_Project_Phenorob/CerraData-4MM/experiment_diff_percentages/weights/frozen_moco_encoder"

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
    echo "Running L2 from MoCo (14ch) with ${PCT}% data"
    echo "========================================="

    python train_l2_from_simclr.py \
        --moco_encoder "$MOCO_ENCODER" \
        --data_dir /home/s52melba/CerraData_Project_Phenorob/CerraData-4MM/dataset_splitted \
        --experiment_name "l2_from_moco_14ch_frozen_${PCT_NAME}percent" \
        --gpu_ids "0" \
        --batch_size 100 \
        --num_epochs 500 \
        --learning_rate 1e-4 \
        --data_percentage $PCT \
        --patience $PATIENCE \
        --num_workers 4 \
        --norm z_score \
        --freeze_encoder \
        --checkpoint_dir "$CHECKPOINT_BASE" \
        --log_dir "$LOG_BASE"

    echo "Completed ${PCT}% at: $(date)"
done

echo ""
echo "=== ALL L2 FROM MOCO (14-CHANNEL) EXPERIMENTS COMPLETED ==="
echo "Finished at: $(date)"
