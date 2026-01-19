#!/bin/bash

# Inference script for CerraData segmentation model
# Usage: bash inference.sh

# Set paths
CHECKPOINT_PATH="/scratch/s52melba/CerraData_Project_Phenorob/checkpoints/baseline_l2_14class_20251013_051810/last.ckpt"
DATA_DIR="/scratch/s52melba/CerraData_Project_Phenorob/kaggle_temp/cerradata_4mm"
N_IMAGES=10
LABEL_LEVEL="L2"

echo "Starting inference..."
# Run inference
python inference.py \
--checkpoint_path "$CHECKPOINT_PATH" \
--data_dir "$DATA_DIR" \
--n_images $N_IMAGES \
--label_level $LABEL_LEVEL

echo "Inference completed!"