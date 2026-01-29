#!/bin/bash
#SBATCH --job-name=test_model
#SBATCH --output=logs/test_model_%j.out
#SBATCH --error=logs/test_model_%j.err

# CHECKPOINT_PATH: Path to your trained model checkpoint (e.g., "./checkpoints/best_model.ckpt")
CHECKPOINT_PATH="./checkpoints/best_model.ckpt"

# LABEL_LEVEL: Either "L1" for 7 classes or "L2" for 14 classes
LABEL_LEVEL="L2"

echo "=== TESTING MODEL ON FULL TEST SET ==="
echo "Starting at: $(date)"

python test_model.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --data_dir ./data_split \
    --label_level "$LABEL_LEVEL" \
    --gpu_ids "0" \
    --batch_size 32 \
    --num_workers 4

echo "Completed at: $(date)"