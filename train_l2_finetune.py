#!/usr/bin/env python3
"""
Fine-tune L2 model using L1 pretrained checkpoint with FROZEN ENCODER
"""
import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='osgeo')
warnings.filterwarnings('ignore', message='Can\'t initialize NVML')

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from datetime import datetime
from torch.utils.data import DataLoader, Subset

# Import MMDataset from CerraData-4MM (Multimodal: MSI+SAR, 14 channels, L2 classes)
from dataset_loader_official.dataset_loader import MMDataset

from model import create_model

def train_l2_finetune(
    l1_checkpoint_path,
    data_dir,
    batch_size=16,
    num_epochs=100,
    learning_rate=1e-4,  # Lower LR for fine-tuning
    num_workers=4,
    gpu_ids=None,
    checkpoint_dir="./checkpoints",
    log_dir="./logs",
    experiment_name=None,
    data_percentage=100,
    patience=20,
    seed=42,
    freeze_encoder=True  # NEW: Option to freeze encoder
):
    """Fine-tune L2 model using L1 pretrained weights with frozen encoder"""

    print("=== Fine-tuning L2 Model from L1 Pretraining ===")
    print(f"L1 checkpoint: {l1_checkpoint_path}")
    print(f"Data directory: {data_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Max epochs: {num_epochs}")
    print(f"GPU IDs: {gpu_ids}")
    print(f"Freeze encoder: {freeze_encoder}")

    # Set seeds for reproducibility
    pl.seed_everything(seed, workers=True)
    print(f"Random seed: {seed}")

    # Handle GPU configuration
    if gpu_ids is None:
        accelerator = 'cpu'
        devices = 1
        strategy = 'auto'
        use_gpu = False
    else:
        accelerator = 'gpu'
        devices = gpu_ids
        strategy = 'ddp' if len(gpu_ids) > 1 else 'auto'
        use_gpu = True

    print(f"Accelerator: {accelerator}")
    print(f"Devices: {devices}")
    print(f"Strategy: {strategy}")

    # Create data loaders using CerraData-4MM's MMDataset (MSI+SAR, 14 channels)
    print(f"\nLoading CerraData-4MM datasets (L2: 14 classes, Multimodal)...")

    # Load datasets on CPU (PyTorch Lightning will move to GPU automatically)
    train_dataset_full = MMDataset(dir_path=os.path.join(data_dir, 'train'), gpu='cpu', norm='none')
    val_dataset_full = MMDataset(dir_path=os.path.join(data_dir, 'val'), gpu='cpu', norm='none')
    test_dataset = MMDataset(dir_path=os.path.join(data_dir, 'test'), gpu='cpu', norm='none')

    # Apply data percentage to train and val (with seed for reproducibility)
    if data_percentage < 100:
        # Train subset
        train_size = max(10, int(round(len(train_dataset_full) * data_percentage / 100)))
        train_indices = torch.randperm(len(train_dataset_full), generator=torch.Generator().manual_seed(seed))[:train_size]
        train_dataset = Subset(train_dataset_full, train_indices.tolist())

        # Val subset
        val_size = max(10, int(round(len(val_dataset_full) * data_percentage / 100)))
        val_indices = torch.randperm(len(val_dataset_full), generator=torch.Generator().manual_seed(seed))[:val_size]
        val_dataset = Subset(val_dataset_full, val_indices.tolist())

        print(f"Using {data_percentage}% of data (seed={seed}):")
        print(f"  Train: {len(train_dataset)} / {len(train_dataset_full)} samples")
        print(f"  Val: {len(val_dataset)} / {len(val_dataset_full)} samples")
    else:
        train_dataset = train_dataset_full
        val_dataset = val_dataset_full
        print(f"Using 100% of data:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")

    print(f"  Test: {len(test_dataset)} samples (always 100%)")

    # Create DataLoaders with optimizations
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              persistent_workers=True if num_workers > 0 else False,
                              pin_memory=True if use_gpu else False,
                              prefetch_factor=4 if num_workers > 0 else None)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            persistent_workers=True if num_workers > 0 else False,
                            pin_memory=True if use_gpu else False,
                            prefetch_factor=4 if num_workers > 0 else None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             persistent_workers=True if num_workers > 0 else False,
                             pin_memory=True if use_gpu else False,
                             prefetch_factor=4 if num_workers > 0 else None)

    print(f"\nDataLoader batches:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")

    # Load L1 pretrained model
    print(f"\nLoading L1 pretrained model from: {l1_checkpoint_path}")
    l1_model = create_model(
        in_channels=14,  # Multimodal: 12 MSI + 2 SAR
        num_classes=7,   # L1 has 7 classes
        encoder_name="resnet34",
        learning_rate=learning_rate
    )

    # Load checkpoint weights
    checkpoint = torch.load(l1_checkpoint_path, map_location='cpu')
    l1_model.load_state_dict(checkpoint['state_dict'])
    print("L1 checkpoint loaded successfully")

    # Create new L2 model with same encoder
    print("Creating L2 model with pretrained encoder...")
    l2_model = create_model(
        in_channels=14,  # Multimodal: 12 MSI + 2 SAR
        num_classes=14,  # L2 has 14 classes
        encoder_name="resnet34",
        learning_rate=learning_rate
    )

    # Transfer encoder weights from L1 to L2
    print("Transferring encoder weights from L1 to L2...")
    l2_model.model.encoder.load_state_dict(l1_model.model.encoder.state_dict())

    # Transfer decoder weights from L1 to L2 (compatible parts)
    print("Transferring decoder weights from L1 to L2...")
    l2_model.model.decoder.load_state_dict(l1_model.model.decoder.state_dict())

    # Only the segmentation head (final layer) will be randomly initialized for 14 classes
    print("Segmentation head randomly initialized for 14 classes")

    # FREEZE ENCODER - Only train decoder and segmentation head
    if freeze_encoder:
        print("\n🔒 FREEZING ENCODER - Only decoder will be trained")
        for param in l2_model.model.encoder.parameters():
            param.requires_grad = False

        # Count trainable vs frozen parameters
        trainable_params = sum(p.numel() for p in l2_model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in l2_model.parameters() if not p.requires_grad)
        total_params = trainable_params + frozen_params

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        print(f"Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    else:
        print("\nEncoder NOT frozen - Full model will be trained")
        print(f"L2 Model parameters: {sum(p.numel() for p in l2_model.parameters()):,}")

    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Setup callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name is None:
        experiment_name = f"l2_finetune_14class_frozenenc_{timestamp}" if freeze_encoder else f"l2_finetune_14class_{timestamp}"
    else:
        experiment_name = f"{experiment_name}_{timestamp}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(checkpoint_dir, experiment_name),
        filename='{epoch}-{val_loss:.4f}-{val_f1_macro:.4f}',
        monitor='val_f1_macro',
        mode='max',
        save_top_k=3,
        save_last=True,
        verbose=True
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        verbose=True,
        mode='min'
    )

    # Setup logger
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=experiment_name,
        version=None
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=9,
        val_check_interval=1.0,
        precision=16 if use_gpu else 32,  # Use mixed precision on GPU
        deterministic=False,  # Disabled due to CUDA cross_entropy determinism issue
        sync_batchnorm=True if (use_gpu and len(devices) > 1) else False  # Sync batch norm for multi-GPU
    )

    print(f"\nStarting L2 fine-tuning...")
    print(f"Experiment: {experiment_name}")
    print(f"Logs: {logger.log_dir}")
    print(f"Checkpoints: {checkpoint_callback.dirpath}")

    # Train model
    trainer.fit(l2_model, train_loader, val_loader)

    # Test model
    print("\nTesting best L2 fine-tuned model...")
    trainer.test(l2_model, test_loader, ckpt_path='best')

    # Save final results
    results_file = os.path.join(checkpoint_callback.dirpath, "training_summary.txt")
    with open(results_file, 'w') as f:
        f.write(f"L2 Fine-tuning Model Training Summary\n")
        f.write(f"====================================\n\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Model: U-Net with ResNet34 encoder (L1 pretrained → L2 fine-tuned)\n")
        f.write(f"Encoder: {'FROZEN' if freeze_encoder else 'TRAINABLE'}\n")
        f.write(f"Task: 14-class semantic segmentation (L2 labels)\n")
        f.write(f"Pretraining: L1 checkpoint from {l1_checkpoint_path}\n")
        f.write(f"Dataset: CerraData-4MM\n")
        f.write(f"Training samples: {len(train_loader.dataset)}\n")
        f.write(f"Validation samples: {len(val_loader.dataset)}\n")
        f.write(f"Test samples: {len(test_loader.dataset)}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Random seed: {seed}\n")
        f.write(f"Total epochs: {trainer.current_epoch + 1}\n")
        f.write(f"Best checkpoint: {checkpoint_callback.best_model_path}\n")
        f.write(f"Final validation loss: {trainer.callback_metrics.get('val_loss', 'N/A')}\n")
        f.write(f"Final validation F1: {trainer.callback_metrics.get('val_f1_macro', 'N/A')}\n")
        f.write(f"Final validation IoU: {trainer.callback_metrics.get('val_iou_macro', 'N/A')}\n")

    print(f"\nL2 fine-tuning completed!")
    print(f"Results saved to: {results_file}")
    print(f"Best model: {checkpoint_callback.best_model_path}")

    return l2_model, trainer, checkpoint_callback.best_model_path

def main():
    parser = argparse.ArgumentParser(description='Fine-tune L2 model from L1 checkpoint with frozen encoder')
    parser.add_argument('--l1_checkpoint', type=str, required=True,
                        help='Path to L1 pretrained checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate (lower for fine-tuning)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help='GPU IDs to use (e.g., "0" or "0,1,2,3" for multi-GPU)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory to save logs')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Custom experiment name (timestamp will be appended)')
    parser.add_argument('--data_percentage', type=float, default=100,
                        help='Percentage of data to use (0.1-100, accepts decimals)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (number of epochs)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--freeze_encoder', action='store_true', default=True,
                        help='Freeze encoder (only train decoder). Default: True')
    parser.add_argument('--no_freeze_encoder', dest='freeze_encoder', action='store_false',
                        help='Do NOT freeze encoder (train full model)')

    args = parser.parse_args()

    # Check if L1 checkpoint exists
    if not os.path.exists(args.l1_checkpoint):
        print(f"Error: L1 checkpoint '{args.l1_checkpoint}' not found!")
        print("Please train L1 model first with: python train_l1_baseline.py")
        return

    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found!")
        return

    # Parse GPU IDs
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]

    # Fine-tune L2 model
    model, trainer, best_model_path = train_l2_finetune(
        l1_checkpoint_path=args.l1_checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        gpu_ids=gpu_ids,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        experiment_name=args.experiment_name,
        data_percentage=args.data_percentage,
        patience=args.patience,
        seed=args.seed,
        freeze_encoder=args.freeze_encoder
    )

if __name__ == "__main__":
    main()
