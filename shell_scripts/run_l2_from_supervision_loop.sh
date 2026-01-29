#!/bin/bash
#SBATCH --job-name=l2_from_ssl_loop
#SBATCH --output=logs/l2_from_ssl_loop_%j.out
#SBATCH --error=logs/l2_from_ssl_loop_%j.err

# UPDATE THIS PATH after SimCLR/MoCo pretraining completes!
# The encoder will be saved in: checkpoints_data_splitted/simclr_pretrain_resnet34_TIMESTAMP/encoder_final.pth
MOCO_ENCODER="./checkpoints_data_splitted/moco_pretrain_resnet34_v2_more_aggressive_20251218_114641/encoder_final.pth"

echo "=== L2 FINE-TUNING FROM SELF-SUPERVISED PRETRAINING - MULTIPLE DATA PERCENTAGES ==="
echo "Using encoder: $MOCO_ENCODER"
echo "Starting at: $(date)"

# Define percentages to test
PERCENTAGES="0.5 1 2.5 3.5 5 10 25 50"

# Early stopping patience
PATIENCE=30

# Organized directories for scaling experiments
LOG_BASE="./logs_scaling_experiments/self_supervised"
CHECKPOINT_BASE="./checkpoints_scaling_experiments/self_supervised"

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
    echo "Running L2 from Self-Supervision with ${PCT}% data"
    echo "========================================="

    python train_l2_from_simclr.py \
        --moco_encoder "$MOCO_ENCODER" \
        --data_dir /home/s52melba/CerraData_Project_Phenorob/cerradata_splitted \
        --experiment_name "l2_from_simclr_${PCT_NAME}percent" \
        --gpu_ids "0" \
        --batch_size 100 \
        --num_epochs 300 \
        --learning_rate 1e-4 \
        --data_percentage $PCT \
        --patience $PATIENCE \
        --num_workers 4 \
        --checkpoint_dir "$CHECKPOINT_BASE" \
        --log_dir "$LOG_BASE"

    echo "Completed ${PCT}% at: $(date)"
done

echo ""
echo "=== ALL L2 FROM SELF-SUPERVISION EXPERIMENTS COMPLETED ==="
echo "Finished at: $(date)"
