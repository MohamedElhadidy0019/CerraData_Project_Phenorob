#!/bin/bash
# Setup and run script for CerraData Project

echo "=== CerraData Project Setup and Training ==="

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Download dataset
echo "Downloading CerraData-4MM dataset..."
python download_data.py

# Test dataset loading
echo "Testing dataset loading..."
python dataset.py

# Test model creation
echo "Testing model creation..."
python model.py

# Train baseline model
echo "Starting baseline training (random initialization on 14-class segmentation)..."
python train_baseline.py --batch_size 8 --num_epochs 50 --learning_rate 1e-3

echo "=== Setup and training complete! ==="
echo "Check ./logs for training logs and ./checkpoints for saved models"