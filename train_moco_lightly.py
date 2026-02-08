#!/usr/bin/env python3
"""
Self-supervised pretraining with MoCo v2 (Momentum Contrast v2)
Uses lightly's simplified approach with memory bank and utility functions
Trains encoder on unlabeled multimodal images (MSI+SAR, 14 channels, train split only)
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
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import copy

import segmentation_models_pytorch as smp
from lightly.models.modules import MoCoProjectionHead
from lightly.loss import NTXentLoss
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule

import kornia.augmentation as K

# Import MMDataset from CerraData-4MM
from dataset_loader_official.dataset_loader import MMDataset


class MultispectralSimCLRTransform:
    """
    Strong augmentations for 14-channel multimodal satellite imagery (MSI+SAR).
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
        max_bands_drop: int = 4,          # Max bands to drop at once (for 14 channels)
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
        This is the multimodal equivalent of color jitter.

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
        Works per-channel for multimodal data.
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

        # 2. Spectral augmentations (multimodal-specific)
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
    Even stronger augmentations for 14-channel multimodal imagery.
    Use this for MoCo training to prevent dimensional collapse.
    """

    def __init__(self, img_size: int = 128):
        super().__init__(
            img_size=img_size,
            crop_scale=(0.08, 1.0),      # Very aggressive crops (like ImageNet)
            band_drop_prob=0.15,
            max_bands_drop=5,             # Drop up to 5 of 14 bands (~36%)
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


class UnlabeledMMDataset(Dataset):
    """
    Wrapper around MMDataset for self-supervised MoCo training.
    Returns only images (14 channels: MSI+SAR), ignores masks.
    """

    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Path to dataset directory (e.g., dataset_splitted)
            transform: Transform for creating augmented views
        """
        # Use MMDataset to load 14-channel multimodal data from train split
        self.dataset = MMDataset(
            dir_path=os.path.join(data_dir, 'train'),
            gpu='cpu',  # Load on CPU, PyTorch Lightning will move to GPU
            norm='1to1'
        )
        self.transform = transform
        print(f"Loaded {len(self.dataset)} unlabeled training images (14 channels: MSI+SAR)")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns:
            If transform is provided: tuple of (view1, view2) - augmented views
            Otherwise: stacked_img (14, H, W)
        """
        stacked_img, _, _ = self.dataset[idx]  # Get image, ignore semantic_mask and edge_mask

        # Apply augmentations (returns tuple of 2 views for contrastive learning)
        if self.transform:
            return self.transform(stacked_img)
        return stacked_img


class MoCoModel(pl.LightningModule):
    """
    MoCo v2 model using lightly's simplified approach
    - Uses NTXentLoss with memory_bank_size (automatic queue management)
    - Uses lightly's update_momentum() helper
    - Uses cosine schedule for momentum coefficient
    """

    def __init__(
        self,
        in_channels=14,  # 12 MSI + 2 SAR
        encoder_name="resnet34",
        learning_rate=0.03,
        weight_decay=1e-4,
        temperature=0.2,
        projection_dim=128,
        hidden_dim=2048,
        memory_bank_size=4096
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Create query encoder (updated by backprop)
        base_unet_q = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=14  # Doesn't matter, we only use encoder
        )
        self.backbone = base_unet_q.encoder

        # MoCo projection head for query
        self.projection_head = MoCoProjectionHead(
            input_dim=512,  # ResNet34 output
            hidden_dim=hidden_dim,
            output_dim=projection_dim
        )

        # Create momentum encoder (key encoder, updated by EMA)
        base_unet_k = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=14
        )
        self.backbone_momentum = base_unet_k.encoder

        # MoCo projection head for key
        self.projection_head_momentum = MoCoProjectionHead(
            input_dim=512,
            hidden_dim=hidden_dim,
            output_dim=projection_dim
        )

        # Initialize momentum networks as copies
        self.backbone_momentum.load_state_dict(self.backbone.state_dict())
        self.projection_head_momentum.load_state_dict(self.projection_head.state_dict())

        # Deactivate gradients for momentum networks
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # NTXentLoss with memory bank (automatic queue management!)
        self.criterion = NTXentLoss(
            temperature=temperature,
            memory_bank_size=(memory_bank_size, projection_dim)
        )

        print(f"MoCo v2 model initialized:")
        print(f"  Input channels: {in_channels} (12 MSI + 2 SAR)")
        print(f"  Encoder: {encoder_name} (output dim: 512)")
        print(f"  Projection: 512 → {hidden_dim} → {projection_dim}")
        print(f"  Temperature: {temperature}")
        print(f"  Memory bank size: {memory_bank_size}")

    def forward(self, x):
        """Forward pass through query encoder"""
        features = self.backbone(x)
        h = features[-1]  # (B, 512, H', W')
        h = nn.functional.adaptive_avg_pool2d(h, 1)  # (B, 512, 1, 1)
        h = h.flatten(1)  # (B, 512)
        z = self.projection_head(h)  # (B, projection_dim)
        return z

    def forward_momentum(self, x):
        """Forward pass through momentum encoder (no gradients)"""
        features = self.backbone_momentum(x)
        h = features[-1]
        h = nn.functional.adaptive_avg_pool2d(h, 1)
        h = h.flatten(1)
        z = self.projection_head_momentum(h)
        return z

    def training_step(self, batch, batch_idx):
        """Training step with MoCo contrastive loss"""
        # Unpack the two augmented views
        (x_q, x_k) = batch

        # Update momentum encoder using cosine schedule
        # Momentum increases from ~0.996 to 1.0 over training
        momentum = cosine_schedule(
            step=self.current_epoch,
            max_steps=self.trainer.max_epochs,
            start_value=0.996,
            end_value=1.0
        )
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)

        # Forward pass through query encoder (with gradients)
        q = self.forward(x_q)

        # Forward pass through key encoder (no gradients)
        with torch.no_grad():
            k = self.forward_momentum(x_k)

        # Compute MoCo loss
        # NTXentLoss with memory bank handles queue automatically
        loss = self.criterion(q, k)

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'],
                 on_step=False, on_epoch=True)
        self.log('momentum', momentum, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """
        Configure optimizer and scheduler (MoCo v2 style)
        - Optimizer: SGD with momentum=0.9
        - Scheduler: Cosine annealing with warmup
        """
        # Only optimize query encoder and projection head
        params = list(self.backbone.parameters()) + list(self.projection_head.parameters())

        optimizer = torch.optim.SGD(
            params,
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=self.weight_decay
        )

        # Get total epochs
        max_epochs = self.trainer.max_epochs if self.trainer else 200
        warmup_epochs = 10

        # Warmup scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-4,
            end_factor=1.0,
            total_iters=warmup_epochs
        )

        # Cosine annealing
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs - warmup_epochs,
            eta_min=1e-8
        )

        # Combine schedulers
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
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
        """Save only query encoder weights (not projection head or momentum encoder)"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.backbone.state_dict(), path)
        print(f"Encoder weights saved to: {path}")


def train_moco(
    data_dir,
    batch_size=256,
    num_epochs=200,
    learning_rate=0.03,
    temperature=0.2,
    projection_dim=128,
    memory_bank_size=4096,
    num_workers=4,
    prefetch_factor=2,
    gpu_ids=None,
    checkpoint_dir="./checkpoints_data_splitted",
    log_dir="./logs_splitted",
    experiment_name=None
):
    """Train MoCo v2 model on unlabeled training images"""

    print("=== MoCo v2 Self-Supervised Pretraining (14-channel Multimodal) ===")
    print(f"Data directory: {data_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Max epochs: {num_epochs}")
    print(f"Temperature: {temperature}")
    print(f"Memory bank size: {memory_bank_size}")
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

    # Use strong augmentations for MoCo training (14 channels)
    transform = MultispectralSimCLRTransformStrong()

    # Create unlabeled dataset using MMDataset wrapper
    print("\nCreating unlabeled dataset (train split only)...")
    train_dataset = UnlabeledMMDataset(
        data_dir=data_dir,
        transform=transform
    )

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_gpu,
        drop_last=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )

    print(f"Train batches: {len(train_loader)}")

    # Create MoCo v2 model (14 channels)
    print("\nCreating MoCo v2 model...")
    model = MoCoModel(
        in_channels=14,  # 12 MSI + 2 SAR
        encoder_name="resnet34",
        learning_rate=learning_rate,
        temperature=temperature,
        projection_dim=projection_dim,
        memory_bank_size=memory_bank_size
    )

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Setup callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name is None:
        experiment_name = f"moco_pretrain_14ch_{timestamp}"
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
        sync_batchnorm=True if (use_gpu and len(devices) > 1) else False,
        gradient_clip_val=1.0  # Prevent NaN from gradient explosion
    )

    print(f"\nStarting MoCo v2 pretraining...")
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
        f.write(f"MoCo v2 Self-Supervised Pretraining Summary\n")
        f.write(f"============================================\n\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Model: U-Net ResNet34 encoder with MoCo v2\n")
        f.write(f"Task: Self-supervised contrastive learning\n")
        f.write(f"Dataset: CerraData-4MM (unlabeled training images)\n")
        f.write(f"Input: 14 channels (12 MSI + 2 SAR)\n")
        f.write(f"Training samples: {len(train_dataset)}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Temperature: {temperature}\n")
        f.write(f"Memory bank size: {memory_bank_size}\n")
        f.write(f"Projection dim: {projection_dim}\n")
        f.write(f"Total epochs: {trainer.current_epoch + 1}\n")
        f.write(f"Best checkpoint: {checkpoint_callback.best_model_path}\n")
        f.write(f"Encoder saved: {encoder_path}\n")
        f.write(f"Final train loss: {trainer.callback_metrics.get('train_loss_epoch', 'N/A')}\n")

    print(f"\nMoCo v2 pretraining completed!")
    print(f"Results saved to: {results_file}")
    print(f"Encoder weights: {encoder_path}")

    return model, trainer, encoder_path


def main():
    parser = argparse.ArgumentParser(description='MoCo v2 self-supervised pretraining')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training (default: 256)')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.03,
                        help='Learning rate (default: 0.03 for batch_size=256)')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='Temperature for contrastive loss (default: 0.2)')
    parser.add_argument('--projection_dim', type=int, default=128,
                        help='Projection head output dimension')
    parser.add_argument('--memory_bank_size', type=int, default=4096,
                        help='Size of the memory bank (queue) for negative samples')
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

    # Train MoCo v2
    model, trainer, encoder_path = train_moco(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        projection_dim=args.projection_dim,
        memory_bank_size=args.memory_bank_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        gpu_ids=gpu_ids,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        experiment_name=args.experiment_name
    )


if __name__ == "__main__":
    main()
