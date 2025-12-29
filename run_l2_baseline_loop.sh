#!/bin/bash
#SBATCH --job-name=l2_baseline_loop
#SBATCH --output=logs/l2_baseline_loop_%j.out
#SBATCH --error=logs/l2_baseline_loop_%j.err

echo "=== L2 BASELINE EXPERIMENT - MULTIPLE DATA PERCENTAGES ==="
echo "Starting at: $(date)"

# Define percentages to test
PERCENTAGES="0.5 1 2.5 3.5 5 10 25 50"

# Early stopping patience
PATIENCE=30

# Organized directories for scaling experiments
LOG_BASE="./logs_scaling_experiments/baseline"
CHECKPOINT_BASE="./checkpoints_scaling_experiments/baseline"

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
    echo "Running L2 Baseline with ${PCT}% data"
    echo "========================================="

    python train_baseline.py \
        --data_dir /home/s52melba/CerraData_Project_Phenorob/cerradata_splitted \
        --label_level L2 \
        --experiment_name "l2_baseline_14classes_${PCT_NAME}percent" \
        --gpu_ids "0" \
        --batch_size 100 \
        --num_epochs 300 \
        --learning_rate 1e-3 \
        --checkpoint_dir "$CHECKPOINT_BASE" \
        --log_dir "$LOG_BASE" \
        --data_percentage $PCT \
        --patience $PATIENCE

    echo "Completed ${PCT}% at: $(date)"
done

echo ""
echo "=== ALL L2 BASELINE EXPERIMENTS COMPLETED ==="
echo "Finished at: $(date)"
