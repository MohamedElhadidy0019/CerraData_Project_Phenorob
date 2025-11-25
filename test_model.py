#!/usr/bin/env python3
"""
Test trained model on the full test set with same data splits as training
"""
import os
import torch
import pytorch_lightning as pl
import argparse
import numpy as np
from sklearn.metrics import f1_score, jaccard_score, accuracy_score

from dataset import CerraDataset
from torch.utils.data import DataLoader
from model import UNetSegmentation

def test_model(
    checkpoint_path,
    data_dir,
    label_level='L2',
    batch_size=16,
    num_workers=4,
    gpu_ids=None
):
    """Test model on full test set using same splits as training"""
    
    print("=== Testing Model on Full Test Set ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data directory: {data_dir}")
    print(f"Label level: {label_level}")
    print(f"Batch size: {batch_size}")
    print(f"GPU IDs: {gpu_ids}")
    
    # Handle GPU configuration
    if gpu_ids is None:
        accelerator = 'cpu'
        devices = 1
        use_gpu = False
    else:
        accelerator = 'gpu'
        devices = gpu_ids
        use_gpu = True
    
    print(f"Accelerator: {accelerator}")
    print(f"Devices: {devices}")
    
    # Create test dataset with SAME splitting logic as training
    print("\nCreating test dataset...")
    test_dataset = CerraDataset(
        data_dir=data_dir, 
        split='test', 
        label_level=label_level,
        # Using same default parameters as training to ensure identical splits
        train_ratio=0.7, 
        val_ratio=0.15, 
        test_ratio=0.15, 
        random_state=42
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Create test data loader (full test set, no percentage sampling)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=use_gpu
    )
    
    print(f"Test batches: {len(test_loader)}")
    
    # Load trained model
    print(f"\nLoading model from: {checkpoint_path}")
    model = UNetSegmentation.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Create trainer for testing
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
        enable_checkpointing=False
    )
    
    print("\nRunning inference on test set...")
    
    # Collect all predictions and labels
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            if use_gpu:
                images = images.cuda()
                labels = labels.cuda()
            
            # Forward pass
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            
            # Store predictions and labels
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Flatten for metrics calculation
    all_preds_flat = all_preds.flatten()
    all_labels_flat = all_labels.flatten()
    
    print(f"Total test pixels: {len(all_preds_flat):,}")
    print(f"Unique labels in test: {sorted(np.unique(all_labels_flat))}")
    print(f"Unique predictions: {sorted(np.unique(all_preds_flat))}")
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels_flat, all_preds_flat)
    f1_macro = f1_score(all_labels_flat, all_preds_flat, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels_flat, all_preds_flat, average='weighted', zero_division=0)
    iou_macro = jaccard_score(all_labels_flat, all_preds_flat, average='macro', zero_division=0)
    
    # Per-class metrics
    f1_per_class = f1_score(all_labels_flat, all_preds_flat, average=None, zero_division=0)
    iou_per_class = jaccard_score(all_labels_flat, all_preds_flat, average=None, zero_division=0)
    
    print("\n" + "="*50)
    print("TEST SET RESULTS")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    print(f"IoU Macro: {iou_macro:.4f}")
    
    print(f"\nPer-class F1 scores:")
    for i, f1 in enumerate(f1_per_class):
        print(f"  Class {i}: {f1:.4f}")
    
    print(f"\nPer-class IoU scores:")
    for i, iou in enumerate(iou_per_class):
        print(f"  Class {i}: {iou:.4f}")
    
    # Save results to file
    results_dir = os.path.dirname(checkpoint_path)
    results_file = os.path.join(results_dir, "test_results.txt")
    
    with open(results_file, 'w') as f:
        f.write("Test Set Evaluation Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model: {checkpoint_path}\n")
        f.write(f"Label Level: {label_level}\n")
        f.write(f"Test Samples: {len(test_dataset)}\n")
        f.write(f"Total Test Pixels: {len(all_preds_flat):,}\n\n")
        f.write("Overall Metrics:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Macro: {f1_macro:.4f}\n")
        f.write(f"F1 Weighted: {f1_weighted:.4f}\n")
        f.write(f"IoU Macro: {iou_macro:.4f}\n\n")
        f.write("Per-class F1 scores:\n")
        for i, f1 in enumerate(f1_per_class):
            f.write(f"Class {i}: {f1:.4f}\n")
        f.write("\nPer-class IoU scores:\n")
        for i, iou in enumerate(iou_per_class):
            f.write(f"Class {i}: {iou:.4f}\n")
    
    print(f"\nResults saved to: {results_file}")
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'iou_macro': iou_macro,
        'f1_per_class': f1_per_class,
        'iou_per_class': iou_per_class
    }

def main():
    parser = argparse.ArgumentParser(description='Test trained model on full test set')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to dataset directory')
    parser.add_argument('--label_level', type=str, default='L2', choices=['L1', 'L2'],
                        help='Label level: L1 (7 classes) or L2 (14 classes)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help='GPU IDs to use (e.g., "0" or "0,1,2,3")')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint '{args.checkpoint}' not found!")
        return
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found!")
        return
    
    # Parse GPU IDs
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    
    # Test model
    results = test_model(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        label_level=args.label_level,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gpu_ids=gpu_ids
    )

if __name__ == "__main__":
    main()