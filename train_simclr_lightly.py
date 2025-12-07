#!/usr/bin/env python3
"""
Self-supervised pretraining with SimCLR using lightly library
Trains encoder on unlabeled MSI images (train split only)
"""
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import rasterio
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp
from lightly.models.modules import SimCLRProjectionHead
from lightly.loss import NTXentLoss
from lightly.transforms import SimCLRTransform


class UnlabeledMSIDataset(Dataset):
    """Dataset for loading unlabeled MSI images (train split only)"""

    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Path to dataset directory (e.g., cerradata_splitted)
            transform: lightly transform for creating augmented views
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "train" / "images"
        self.transform = transform

        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")

        # Get all training image files
        self.image_files = list(self.images_dir.glob("*.tif"))
        self.image_files.sort()

        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {self.images_dir}")

        print(f"Loaded {len(self.image_files)} unlabeled training images")

        # Compute global statistics across entire training set
        print("Computing global per-channel statistics...")
        self.global_min, self.global_max = self._compute_global_stats()
        print(f"Global min per channel: {self.global_min}")
        print(f"Global max per channel: {self.global_max}")

    def _compute_global_stats(self):
        """Compute global min/max per channel across all training images"""
        num_channels = 12
        channel_mins = [float('inf')] * num_channels
        channel_maxs = [float('-inf')] * num_channels

        print(f"Scanning {len(self.image_files)} images for statistics...")
        for idx, img_path in enumerate(self.image_files):
            if idx % 100 == 0:
                print(f"  Processed {idx}/{len(self.image_files)} images...")

            with rasterio.open(img_path) as src:
                image = src.read()  # Shape: (12, H, W)
                image = image.astype(np.float32)

            # Update min/max for each channel
            for ch in range(num_channels):
                ch_min = image[ch].min()
                ch_max = image[ch].max()
                channel_mins[ch] = min(channel_mins[ch], ch_min)
                channel_maxs[ch] = max(channel_maxs[ch], ch_max)

        print(f"  Processed {len(self.image_files)}/{len(self.image_files)} images. Done!")
        return channel_mins, channel_maxs

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load multispectral image (12 channels)
        img_path = self.image_files[idx]
        with rasterio.open(img_path) as src:
            image = src.read()  # Shape: (12, H, W)
            image = image.astype(np.float32)

        # Per-channel normalization using global statistics
        normalized_image = np.zeros_like(image)
        for ch in range(12):
            ch_min = self.global_min[ch]
            ch_max = self.global_max[ch]
            normalized_image[ch] = (image[ch] - ch_min) / (ch_max - ch_min + 1e-8)

        # Convert to torch tensor
        image = torch.from_numpy(normalized_image)

        # Apply SimCLR transforms if provided
        if self.transform:
            image = self.transform(image)

        return image


class SimCLRModel(pl.LightningModule):
    """SimCLR model using lightly library"""

    def __init__(
        self,
        in_channels=12,
        encoder_name="resnet34",
        learning_rate=1e-3,
        weight_decay=1e-4,
        temperature=0.5,
        projection_dim=128,
        hidden_dim=256
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Create encoder (same as U-Net encoder)
        base_unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,  # Random initialization
            in_channels=in_channels,
            classes=14  # Doesn't matter, we only use encoder
        )
        self.encoder = base_unet.encoder

        # Get encoder output dimension
        # For ResNet34, the last layer outputs 512 channels
        self.encoder_dim = 512

        # SimCLR projection head (used only during pretraining)
        self.projection_head = SimCLRProjectionHead(
            input_dim=self.encoder_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim
        )

        # Contrastive loss
        self.criterion = NTXentLoss(temperature=temperature)

        print(f"SimCLR model initialized:")
        print(f"  Encoder: {encoder_name} (output dim: {self.encoder_dim})")
        print(f"  Projection: {self.encoder_dim} → {hidden_dim} → {projection_dim}")
        print(f"  Temperature: {temperature}")

    def forward(self, x):
        """Forward pass through encoder and projection head"""
        # Encode
        features = self.encoder(x)

        # Get last feature map
        h = features[-1]  # Shape: (B, 512, H', W')

        # Global average pooling
        h = nn.functional.adaptive_avg_pool2d(h, 1)  # Shape: (B, 512, 1, 1)
        h = h.flatten(1)  # Shape: (B, 512)

        # Project
        z = self.projection_head(h)  # Shape: (B, projection_dim)

        return z

    def training_step(self, batch, batch_idx):
        """Training step with SimCLR contrastive loss"""
        # batch contains two views: (view0, view1)
        # Each view has shape: (B, C, H, W)
        (x0, x1) = batch[0]

        # Get representations for both views
        z0 = self(x0)
        z1 = self(x1)

        # Compute contrastive loss
        loss = self.criterion(z0, z1)

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'],
                 on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss_epoch",
            },
        }

    def save_encoder(self, path):
        """Save only encoder weights (not projection head)"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.encoder.state_dict(), path)
        print(f"Encoder weights saved to: {path}")


def train_simclr(
    data_dir,
    batch_size=100,
    num_epochs=200,
    learning_rate=1e-3,
    temperature=0.5,
    projection_dim=128,
    num_workers=4,
    gpu_ids=None,
    checkpoint_dir="./checkpoints_data_splitted",
    log_dir="./logs_splitted",
    experiment_name=None
):
    """Train SimCLR model on unlabeled training images"""

    print("=== SimCLR Self-Supervised Pretraining ===")
    print(f"Data directory: {data_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Max epochs: {num_epochs}")
    print(f"Temperature: {temperature}")
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

    # Create SimCLR transforms
    # Note: lightly's SimCLRTransform expects input_size as int
    # Disabled color jitter and grayscale as they are RGB-specific and not appropriate for 12-channel multispectral data
    transform = SimCLRTransform(
        input_size=128,  # Your image size
        cj_prob=0.0,     # Color jitter disabled (not for multispectral)
        cj_strength=0.5, # (ignored when cj_prob=0)
        min_scale=0.5,   # Minimum crop scale
        gaussian_blur=0.5,  # Gaussian blur probability
        random_gray_scale=0.0,  # Grayscale disabled (not for multispectral)
        hf_prob=0.5,     # Horizontal flip
        vf_prob=0.5,     # Vertical flip (good for satellite imagery)
    )

    # Create dataset (unlabeled, train split only)
    print("\nCreating unlabeled dataset (train split only)...")
    train_dataset = UnlabeledMSIDataset(
        data_dir=data_dir,
        transform=transform
    )

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Important for contrastive learning
    )

    print(f"Train batches: {len(train_loader)}")

    # Create SimCLR model
    print("\nCreating SimCLR model...")
    model = SimCLRModel(
        in_channels=12,
        encoder_name="resnet34",
        learning_rate=learning_rate,
        temperature=temperature,
        projection_dim=projection_dim
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Setup callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name is None:
        experiment_name = f"simclr_pretrain_{timestamp}"
    else:
        experiment_name = f"{experiment_name}_{timestamp}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(checkpoint_dir, experiment_name),
        filename='{epoch}-{train_loss_epoch:.4f}',
        monitor='train_loss_epoch',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True
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
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=10,
        precision=16 if use_gpu else 32,
        deterministic=False,
        sync_batchnorm=True if (use_gpu and len(devices) > 1) else False
    )

    print(f"\nStarting SimCLR pretraining...")
    print(f"Experiment: {experiment_name}")
    print(f"Logs: {logger.log_dir}")
    print(f"Checkpoints: {checkpoint_callback.dirpath}")

    # Train model
    trainer.fit(model, train_loader)

    # Save encoder weights
    encoder_path = os.path.join(checkpoint_callback.dirpath, "encoder_final.pth")
    model.save_encoder(encoder_path)

    # Save training summary
    results_file = os.path.join(checkpoint_callback.dirpath, "training_summary.txt")
    with open(results_file, 'w') as f:
        f.write(f"SimCLR Self-Supervised Pretraining Summary\n")
        f.write(f"==========================================\n\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Model: U-Net ResNet34 encoder with SimCLR\n")
        f.write(f"Task: Self-supervised contrastive learning\n")
        f.write(f"Dataset: CerraData-4MM (unlabeled training images)\n")
        f.write(f"Training samples: {len(train_dataset)}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Temperature: {temperature}\n")
        f.write(f"Projection dim: {projection_dim}\n")
        f.write(f"Total epochs: {trainer.current_epoch + 1}\n")
        f.write(f"Best checkpoint: {checkpoint_callback.best_model_path}\n")
        f.write(f"Encoder saved: {encoder_path}\n")
        f.write(f"Final train loss: {trainer.callback_metrics.get('train_loss_epoch', 'N/A')}\n")

    print(f"\nSimCLR pretraining completed!")
    print(f"Results saved to: {results_file}")
    print(f"Encoder weights: {encoder_path}")

    return model, trainer, encoder_path


def main():
    parser = argparse.ArgumentParser(description='SimCLR self-supervised pretraining')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature for NT-Xent loss')
    parser.add_argument('--projection_dim', type=int, default=128,
                        help='Projection head output dimension')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help='GPU IDs to use (e.g., "0" or "0,1,2,3")')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_data_splitted',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs_splitted',
                        help='Directory to save logs')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Custom experiment name (timestamp will be appended)')

    args = parser.parse_args()

    # Set seeds for reproducibility
    pl.seed_everything(42, workers=True)

    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found!")
        return

    # Parse GPU IDs
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]

    # Train SimCLR
    model, trainer, encoder_path = train_simclr(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        projection_dim=args.projection_dim,
        num_workers=args.num_workers,
        gpu_ids=gpu_ids,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        experiment_name=args.experiment_name
    )


if __name__ == "__main__":
    main()
