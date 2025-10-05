#!/usr/bin/env python3
"""
Train baseline U-Net model on 14-class segmentation with random initialization
"""
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from datetime import datetime

from dataset import create_data_loaders
from model import create_model

def train_baseline(
    data_dir,
    batch_size=16,
    num_epochs=100,
    learning_rate=1e-3,
    num_workers=4,
    gpu_ids=None,
    checkpoint_dir="./checkpoints",
    log_dir="./logs"
):
    """Train baseline model with random initialization on 14-class segmentation"""
    
    print("=== Training Baseline Model (Random Initialization) ===")
    print(f"Data directory: {data_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Max epochs: {num_epochs}")
    print(f"GPU IDs: {gpu_ids}")
    
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
    
    # Create data loaders for L2 (14-class) labels
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        label_level='L2'  # Use fine-level 14-class labels
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model with random initialization
    print("\nCreating U-Net model...")
    model = create_model(
        in_channels=12,
        num_classes=14,
        encoder_name="resnet34",
        learning_rate=learning_rate
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Model uses random initialization (no pretrained weights)")
    
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"baseline_l2_14class_{timestamp}"
    
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
        patience=15,
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
        log_every_n_steps=50,
        val_check_interval=1.0,
        precision='16-mixed' if use_gpu else 32,  # Use mixed precision on GPU
        deterministic=True,  # For reproducible results
        sync_batchnorm=True if (use_gpu and len(devices) > 1) else False  # Sync batch norm for multi-GPU
    )
    
    print(f"\nStarting training...")
    print(f"Experiment: {experiment_name}")
    print(f"Logs: {logger.log_dir}")
    print(f"Checkpoints: {checkpoint_callback.dirpath}")
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Test model
    print("\nTesting best model...")
    trainer.test(model, test_loader, ckpt_path='best')
    
    # Save final results
    results_file = os.path.join(checkpoint_callback.dirpath, "training_summary.txt")
    with open(results_file, 'w') as f:
        f.write(f"Baseline Model Training Summary\n")
        f.write(f"================================\n\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Model: U-Net with ResNet34 encoder (random initialization)\n")
        f.write(f"Task: 14-class semantic segmentation (L2 labels)\n")
        f.write(f"Dataset: CerraData-4MM\n")
        f.write(f"Training samples: {len(train_loader.dataset)}\n")
        f.write(f"Validation samples: {len(val_loader.dataset)}\n")
        f.write(f"Test samples: {len(test_loader.dataset)}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Total epochs: {trainer.current_epoch + 1}\n")
        f.write(f"Best checkpoint: {checkpoint_callback.best_model_path}\n")
        f.write(f"Final validation loss: {trainer.callback_metrics.get('val_loss', 'N/A')}\n")
        f.write(f"Final validation F1: {trainer.callback_metrics.get('val_f1_macro', 'N/A')}\n")
        f.write(f"Final validation IoU: {trainer.callback_metrics.get('val_iou_macro', 'N/A')}\n")
    
    print(f"\nTraining completed!")
    print(f"Results saved to: {results_file}")
    print(f"Best model: {checkpoint_callback.best_model_path}")
    
    return model, trainer, checkpoint_callback.best_model_path

def main():
    parser = argparse.ArgumentParser(description='Train baseline U-Net model')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help='GPU IDs to use (e.g., "0" or "0,1,2,3" for multi-GPU)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory to save logs')
    
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    pl.seed_everything(42, workers=True)
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found!")
        print("Please run 'python download_data.py' first to download the dataset.")
        return
    
    # Parse GPU IDs
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    
    # Train baseline model
    model, trainer, best_model_path = train_baseline(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        gpu_ids=gpu_ids,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )

if __name__ == "__main__":
    main()