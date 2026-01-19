#!/bin/bash
#SBATCH --job-name=create_splits
#SBATCH --output=logs/create_splits_%j.out
#SBATCH --error=logs/create_splits_%j.err

# SOURCE_DIR: Path to original CerraData dataset
SOURCE_DIR="/home/s52melba/CerraData_Project_Phenorob/kaggle_temp/cerradata_4mm"

# TARGET_DIR: Path where physically split dataset will be created
TARGET_DIR="cerradata_splitted"

echo "=== CREATING PHYSICAL DATASET SPLITS ==="
echo "Starting at: $(date)"

python create_physical_splits.py \
    --source_dir "$SOURCE_DIR" \
    --target_dir "$TARGET_DIR" \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --random_state 42

echo "Completed at: $(date)"