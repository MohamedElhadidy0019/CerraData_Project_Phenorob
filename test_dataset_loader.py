#!/usr/bin/env python3
"""
Test script to verify CerraData-4MM MMDataset loading for both L1 and L2
Prints dataset sizes and sample shapes
"""
import os
import sys
import torch

# Hardcoded dataset path
DATA_DIR = '/home/s52melba/CerraData_Project_Phenorob/CerraData-4MM/dataset_splitted'

# Import MMDataset from CerraData-4MM
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CerraData-4MM', 'CerraData-4MM Experiments', 'util'))
from dataset_loader_7 import MMDataset as MMDataset_L1  # L1 (7 classes)
from dataset_loader import MMDataset as MMDataset_L2  # L2 (14 classes)

def test_l1_dataset():
    """Load and test MMDataset for L1 (7 classes)"""

    print("=== Testing CerraData-4MM MMDataset L1 (7 classes) ===\n")

    # Device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Load datasets
    print("Loading datasets...")
    train_dataset = MMDataset_L1(dir_path=os.path.join(DATA_DIR, 'train'), gpu=device, norm='none')
    val_dataset = MMDataset_L1(dir_path=os.path.join(DATA_DIR, 'val'), gpu=device, norm='none')
    test_dataset = MMDataset_L1(dir_path=os.path.join(DATA_DIR, 'test'), gpu=device, norm='none')

    # Print lengths
    print("\n=== Dataset Sizes ===")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val:   {len(val_dataset)} samples")
    print(f"Test:  {len(test_dataset)} samples")
    print(f"Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)} samples")

    # Get one sample from train
    print("\n=== Sample Shapes (from train split) ===")
    stacked_img, semantic_mask, edge_mask = train_dataset[0]

    print(f"Stacked image (MSI+SAR): {stacked_img.shape}")
    print(f"  - Expected: [14, 128, 128] (14 channels, 128x128 pixels)")
    print(f"  - Channels 0-11: MSI")
    print(f"  - Channels 12-13: SAR")

    print(f"\nSemantic mask (labels):  {semantic_mask.shape}")
    print(f"  - Expected: [128, 128]")
    print(f"  - Unique classes: {torch.unique(semantic_mask).cpu().numpy()}")
    print(f"  - L1 has 7 classes (0-6)")

    print(f"\nEdge mask:               {edge_mask.shape}")
    print(f"  - Expected: [128, 128]")

    # Data types
    print("\n=== Data Types ===")
    print(f"Stacked image: {stacked_img.dtype}")
    print(f"Semantic mask: {semantic_mask.dtype}")
    print(f"Edge mask:     {edge_mask.dtype}")

    # Device check
    print("\n=== Device Location ===")
    print(f"Stacked image: {stacked_img.device}")
    print(f"Semantic mask: {semantic_mask.device}")
    print(f"Edge mask:     {edge_mask.device}")

    print("\n=== L1 Test Completed Successfully! ===")

def test_l2_dataset():
    """Load and test MMDataset for L2 (14 classes)"""

    print("=== Testing CerraData-4MM MMDataset L2 (14 classes) ===\n")

    # Device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Load datasets
    print("Loading datasets...")
    train_dataset = MMDataset_L2(dir_path=os.path.join(DATA_DIR, 'train'), gpu=device, norm='none')
    val_dataset = MMDataset_L2(dir_path=os.path.join(DATA_DIR, 'val'), gpu=device, norm='none')
    test_dataset = MMDataset_L2(dir_path=os.path.join(DATA_DIR, 'test'), gpu=device, norm='none')

    # Print lengths
    print("\n=== Dataset Sizes ===")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val:   {len(val_dataset)} samples")
    print(f"Test:  {len(test_dataset)} samples")
    print(f"Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)} samples")

    # Get one sample from train
    print("\n=== Sample Shapes (from train split) ===")
    stacked_img, semantic_mask, edge_mask = train_dataset[0]

    print(f"Stacked image (MSI+SAR): {stacked_img.shape}")
    print(f"  - Expected: [14, 128, 128] (14 channels, 128x128 pixels)")
    print(f"  - Channels 0-11: MSI")
    print(f"  - Channels 12-13: SAR")

    print(f"\nSemantic mask (labels):  {semantic_mask.shape}")
    print(f"  - Expected: [128, 128]")
    print(f"  - Unique classes: {torch.unique(semantic_mask).cpu().numpy()}")
    print(f"  - L2 has 14 classes (0-13)")

    print(f"\nEdge mask:               {edge_mask.shape}")
    print(f"  - Expected: [128, 128]")

    # Data types
    print("\n=== Data Types ===")
    print(f"Stacked image: {stacked_img.dtype}")
    print(f"Semantic mask: {semantic_mask.dtype}")
    print(f"Edge mask:     {edge_mask.dtype}")

    # Device check
    print("\n=== Device Location ===")
    print(f"Stacked image: {stacked_img.device}")
    print(f"Semantic mask: {semantic_mask.device}")
    print(f"Edge mask:     {edge_mask.device}")

    print("\n=== L2 Test Completed Successfully! ===")

if __name__ == "__main__":
    print("------- start l1 ------\n")
    test_l1_dataset()
    print("\n------- end ------\n")

    print("\n------- start l2 ------\n")
    test_l2_dataset()
    print("\n------- end ------")
