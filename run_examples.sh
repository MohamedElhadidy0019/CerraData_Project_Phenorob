#!/bin/bash
# Example usage of the updated train_baseline.py with new parameters
# 
# New parameters:
# --dataset_percentage: Percentage of dataset to use (1-100), None for full dataset
# --use_imagenet_weights: Use ImageNet pretrained weights (default: random initialization) 
# --random_state: Random seed for reproducibility (default: 42)

echo "=== EXAMPLE TRAINING CONFIGURATIONS ==="

# Example 1: Train on 10% of dataset with random initialization
echo "Example 1: 10% dataset, random weights"
python train_baseline.py \
    --data_dir ./data \
    --label_level L2 \
    --experiment_name "example_10percent_random" \
    --batch_size 16 \
    --num_epochs 20 \
    --dataset_percentage 10 \
    --random_state 42

echo "================================"

# Example 2: Train on 25% of dataset with ImageNet weights  
echo "Example 2: 25% dataset, ImageNet weights"
python train_baseline.py \
    --data_dir ./data \
    --label_level L2 \
    --experiment_name "example_25percent_imagenet" \
    --batch_size 16 \
    --num_epochs 20 \
    --dataset_percentage 25 \
    --use_imagenet_weights \
    --random_state 42

echo "================================"

# Example 3: Train on full dataset with ImageNet weights
echo "Example 3: Full dataset, ImageNet weights"
python train_baseline.py \
    --data_dir ./data \
    --label_level L2 \
    --experiment_name "example_full_imagenet" \
    --batch_size 16 \
    --num_epochs 50 \
    --use_imagenet_weights \
    --random_state 42

echo "================================"

# Example 4: Quick debug run with 5% data
echo "Example 4: Debug - 5% dataset, 2 epochs"
python train_baseline.py \
    --data_dir ./data \
    --label_level L1 \
    --experiment_name "debug_quick" \
    --batch_size 8 \
    --num_epochs 2 \
    --dataset_percentage 5 \
    --random_state 42

echo "================================"

# Example 5: L2 Fine-tuning with 30% dataset (requires L1 checkpoint)
echo "Example 5: L2 Fine-tuning - 30% dataset"
# Uncomment and update L1_CHECKPOINT path when available:
# L1_CHECKPOINT="./checkpoints/l1_pretrain_7classes_TIMESTAMP/best.ckpt"
# python train_l2_finetune.py \
#     --l1_checkpoint "$L1_CHECKPOINT" \
#     --data_dir ./data \
#     --experiment_name "example_finetune_30percent" \
#     --batch_size 16 \
#     --num_epochs 30 \
#     --dataset_percentage 30 \
#     --learning_rate 1e-4 \
#     --random_state 42

echo "L2 fine-tuning example commented out - update L1_CHECKPOINT path first"

echo "=== ALL EXAMPLES COMPLETED ==="