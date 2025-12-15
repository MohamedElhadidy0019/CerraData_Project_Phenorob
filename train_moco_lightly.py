#!/usr/bin/env python3
"""
Self-supervised pretraining with MoCo v2 (Momentum Contrast v2)
Uses lightly's simplified approach with memory bank and utility functions
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
from torch.utils.data import DataLoader
import copy

import segmentation_models_pytorch as smp
from lightly.models.modules import MoCoProjectionHead
from lightly.loss import NTXentLoss
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule

# Import reusable components from SimCLR script
from train_simclr_lightly import UnlabeledMSIDataset, MultispectralSimCLRTransform


class MoCoModel(pl.LightningModule):
    """
    MoCo v2 model using lightly's simplified approach
    - Uses NTXentLoss with memory_bank_size (automatic queue management)
    - Uses lightly's update_momentum() helper
    - Uses cosine schedule for momentum coefficient
    """

    def __init__(
        self,
        in_channels=12,
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
            eta_min=1e-6
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

    print("=== MoCo v2 Self-Supervised Pretraining ===")
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

    # Reuse transform from SimCLR script
    transform = MultispectralSimCLRTransform()

    # Reuse dataset from SimCLR script
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
        drop_last=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )

    print(f"Train batches: {len(train_loader)}")

    # Create MoCo v2 model
    print("\nCreating MoCo v2 model...")
    model = MoCoModel(
        in_channels=12,
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
        experiment_name = f"moco_pretrain_{timestamp}"
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
