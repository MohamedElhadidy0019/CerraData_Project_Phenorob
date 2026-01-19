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

import kornia.augmentation as K
import kornia.filters as KF



class MultispectralSimCLRTransform:
    """
    Strong augmentations for 12-channel multispectral satellite imagery.
    Designed for contrastive learning where we need views that are
    challenging to match but preserve semantic content.
    """

    def __init__(
        self,
        img_size: int = 128,
        # Geometric augmentation params
        crop_scale: tuple = (0.2, 1.0),  # More aggressive cropping
        crop_ratio: tuple = (0.75, 1.33),
        # Spectral augmentation params
        band_drop_prob: float = 0.1,      # Probability to zero out each band
        max_bands_drop: int = 3,          # Max bands to drop at once
        intensity_jitter: float = 0.2,    # Per-band intensity scaling
        noise_std: float = 0.05,          # Gaussian noise std
        # Other params
        blur_prob: float = 0.5,
        solarize_prob: float = 0.2,
        solarize_threshold: float = 0.5,
    ):
        self.img_size = img_size
        self.band_drop_prob = band_drop_prob
        self.max_bands_drop = max_bands_drop
        self.intensity_jitter = intensity_jitter
        self.noise_std = noise_std
        self.solarize_prob = solarize_prob
        self.solarize_threshold = solarize_threshold

        # Geometric augmentations (applied consistently across all bands)
        self.geometric_aug = K.AugmentationSequential(
            K.RandomResizedCrop(
                size=(img_size, img_size),
                scale=crop_scale,
                ratio=crop_ratio,
                p=1.0  # Always crop - this is crucial
            ),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation(degrees=90, p=0.5),
            data_keys=["input"],
            same_on_batch=False
        )

        # Blur augmentation
        self.blur = K.RandomGaussianBlur(
            kernel_size=(3, 7),  # Smaller kernel range
            sigma=(0.1, 2.0),
            p=blur_prob
        )

    def _random_band_drop(self, img: torch.Tensor) -> torch.Tensor:
        """
        Randomly zero out some spectral bands.
        Forces the model to not rely on any single band.

        Args:
            img: (C, H, W) tensor
        """
        if self.band_drop_prob <= 0:
            return img

        C = img.shape[0]
        img = img.clone()

        # Decide how many bands to drop (0 to max_bands_drop)
        n_drop = torch.randint(0, self.max_bands_drop + 1, (1,)).item()

        if n_drop > 0:
            # Randomly select which bands to drop
            drop_indices = torch.randperm(C)[:n_drop]
            img[drop_indices] = 0.0

        return img

    def _intensity_jitter(self, img: torch.Tensor) -> torch.Tensor:
        """
        Per-band intensity scaling (like brightness/contrast for each band).
        This is the MSI equivalent of color jitter.

        Each band gets multiplied by a random factor in [1-jitter, 1+jitter]
        and shifted by a random offset.
        """
        if self.intensity_jitter <= 0:
            return img

        C = img.shape[0]
        img = img.clone()

        # Per-band multiplicative scaling
        scale = 1.0 + (torch.rand(C, 1, 1, device=img.device) * 2 - 1) * self.intensity_jitter

        # Per-band additive shift (smaller magnitude)
        shift = (torch.rand(C, 1, 1, device=img.device) * 2 - 1) * (self.intensity_jitter * 0.5)

        img = img * scale + shift

        return img

    def _add_noise(self, img: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to simulate sensor noise."""
        if self.noise_std <= 0:
            return img

        noise = torch.randn_like(img) * self.noise_std
        return img + noise

    def _solarize(self, img: torch.Tensor) -> torch.Tensor:
        """
        Solarize augmentation: invert pixels above threshold.
        Works per-channel for MSI data.
        """
        if torch.rand(1).item() > self.solarize_prob:
            return img

        # Assuming data is normalized around 0, use absolute threshold
        mask = img > self.solarize_threshold
        img = img.clone()
        img[mask] = -img[mask]  # Invert values above threshold

        return img

    def augment_single(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply full augmentation pipeline to a single image.

        Args:
            img: (C, H, W) tensor, already normalized

        Returns:
            Augmented (C, H, W) tensor
        """
        # 1. Geometric augmentations (add batch dim for kornia)
        img = img.unsqueeze(0)
        img = self.geometric_aug(img)
        img = self.blur(img)
        img = img.squeeze(0)

        # 2. Spectral augmentations (MSI-specific)
        img = self._intensity_jitter(img)
        img = self._random_band_drop(img)
        img = self._add_noise(img)
        img = self._solarize(img)

        return img

    def __call__(self, img: torch.Tensor) -> tuple:
        """
        Generate two augmented views of the input image.

        Args:
            img: (C, H, W) tensor

        Returns:
            Tuple of two augmented views
        """
        view1 = self.augment_single(img)
        view2 = self.augment_single(img)

        return view1, view2


class MultispectralSimCLRTransformStrong(MultispectralSimCLRTransform):
    """
    Even stronger augmentations for when the basic version plateaus.
    Use this if the model still converges too quickly.
    """

    def __init__(self, img_size: int = 128):
        super().__init__(
            img_size=img_size,
            crop_scale=(0.08, 1.0),      # Very aggressive crops (like ImageNet)
            band_drop_prob=0.15,
            max_bands_drop=4,             # Drop up to 4 of 12 bands
            intensity_jitter=0.3,         # Stronger intensity changes
            noise_std=0.08,
            blur_prob=0.5,
            solarize_prob=0.3,
        )

        # Add extra augmentations
        self.cutout = K.RandomErasing(
            p=0.3,
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3),
            value=0.0
        )

    def augment_single(self, img: torch.Tensor) -> torch.Tensor:
        """Apply stronger augmentation pipeline."""
        # Base augmentations
        img = super().augment_single(img)

        # Additional: Random erasing (cutout)
        img = img.unsqueeze(0)
        img = self.cutout(img)
        img = img.squeeze(0)

        return img


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

        # Precompute normalization statistics (computed once, used for all samples)
        _, _, mean, stddev = self._data_info()
        self.mean = np.array(mean).reshape(12, 1, 1).astype(np.float32)
        self.stddev = np.array(stddev).reshape(12, 1, 1).astype(np.float32)

        print(f"Loaded {len(self.image_files)} unlabeled training images")

    
    def _data_info(self):
        # source: https://github.com/ai4luc/CerraData-4MM/blob/main/CerraData-4MM%20Experiments/util/dataset_loader.py
        
        min = [99.78856658935547, 332.65665627643466, 347.161809168756, 331.4168453961611,
                196.89053159952164, 240.9765984416008, 261.34731489419937, 342.50664601475,
                277.87501442432404, 246.40860325098038, 265.9057685136795, 226.23770987987518]
        
        max = [7349.042938232482, 8987.99301147458, 8906.377044677738, 9027.435272216775,
                9090.25390625, 8949.610290527282, 8955.640045166012, 9491.945373535062,
                9026.07144165042, 11857.606872558594, 11817.384948730469, 13970.691894531188]
        
        mean = [1331.2999603920011, 1422.618248839035, 1648.7418838236356, 1811.0396095371318,
                2243.6360604171587, 2862.469356914663, 3158.7246770243464, 3253.5804747400075,
                3464.1887187200564, 3463.5260019211623, 3635.662557047575, 2740.6395025025904]
        
        stddev = [436.04697715189127, 484.32797096427566, 549.125419913045, 741.2668466992163,
                788.8006282648606, 860.9668486457188, 963.2983618801512, 1000.2677835011111,
                1087.111000434025, 1062.9960118331512, 1373.6088616321088, 1125.5168224477407]

        return max, min, mean, stddev


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load multispectral image (12 channels)
        img_path = self.image_files[idx]
        with rasterio.open(img_path) as src:
            image = src.read()  # Shape: (12, H, W)
            image = image.astype(np.float32)

        # Z-score normalization using precomputed statistics
        normalized_image = (image - self.mean) / (self.stddev + 1e-8)

        # Convert to torch tensor
        image = torch.from_numpy(normalized_image).float()

        # Apply SimCLR transforms if provided
        if self.transform:
            image_tuple = self.transform(image)

        return image_tuple


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

        # Unpack the two augmented views from batch
        # batch is a list: [view0_batch, view1_batch]
        # Each view has shape: (B, C, H, W) = (batch_size, 12, 128, 128)
        # batch is usally (image,label) but here it is (image,image_augmented) because of contrastive learning
        (x0, x1) = batch

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
        """Configure optimizer and scheduler with warmup + cosine annealing"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Get total epochs from trainer
        max_epochs = self.trainer.max_epochs if self.trainer else 200
        warmup_epochs = 10

        # Create warmup scheduler (linear warmup from 0 to max_lr)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-4,  # Start at lr * 1e-4
            end_factor=1.0,     # End at lr * 1.0 (full lr)
            total_iters=warmup_epochs
        )

        # Create cosine annealing scheduler (after warmup)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs - warmup_epochs,  # Remaining epochs after warmup
            eta_min=1e-6  # Minimum learning rate
        )

        # Combine warmup + cosine annealing
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]  # Switch at epoch 10
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
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
    prefetch_factor=2,
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

    transform = MultispectralSimCLRTransform()

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
        pin_memory=use_gpu,
        drop_last=True,  # Important for contrastive learning
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
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
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Number of batches to prefetch per worker')
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
        prefetch_factor=args.prefetch_factor,
        gpu_ids=gpu_ids,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        experiment_name=args.experiment_name
    )


if __name__ == "__main__":
    main()
