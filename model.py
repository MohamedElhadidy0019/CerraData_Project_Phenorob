#!/usr/bin/env python3
"""
U-Net model implementation using segmentation-models-pytorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import f1_score, jaccard_score

class UNetSegmentation(pl.LightningModule):
    """U-Net model for semantic segmentation using PyTorch Lightning"""
    
    def __init__(
        self,
        in_channels=12,  # Sentinel-2 has 12 bands
        num_classes=None,  # Must be specified (7 for L1, 14 for L2)
        encoder_name="resnet34",
        learning_rate=1e-3,
        weight_decay=1e-3,
        dropout_rate=0.3,
        class_weights=None
    ):
        super().__init__()
        self.save_hyperparameters()
        
        if num_classes is None:
            raise ValueError("num_classes must be specified (7 for L1, 14 for L2)")
        
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        
        print(f"UNetSegmentation initialized with {num_classes} classes")
        
        # Create U-Net model using segmentation-models-pytorch with NO pretrained weights
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,  # NO pretrained weights - random initialization
            in_channels=in_channels,
            classes=num_classes,
            activation=None  # We'll apply softmax in forward pass
        )
        
        # Add dropout layers for different parts of the network
        self.encoder_dropout = nn.Dropout2d(p=self.dropout_rate)  # For encoder features
        self.decoder_dropout = nn.Dropout2d(p=self.dropout_rate)  # For decoder features
        self.bottleneck_dropout = nn.Dropout2d(p=min(self.dropout_rate * 1.5, 0.5))  # Higher dropout at bottleneck, capped at 0.5
        
        # Loss function with class weights if provided
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Metrics storage
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
    def forward(self, x):
        """Forward pass with integrated dropout"""
        # Use the encoder from the U-Net model
        features = self.model.encoder(x)
        
        # Apply dropout to encoder features during training
        if self.training:
            # Apply dropout to intermediate encoder features (skip connections)
            features = [self.encoder_dropout(feat) if i > 0 else feat 
                       for i, feat in enumerate(features)]
            
            # Apply stronger dropout to bottleneck (last encoder feature)
            features[-1] = self.bottleneck_dropout(features[-1])
        
        # Use the decoder from the U-Net model - pass features as list, not unpacked
        decoder_output = self.model.decoder(features)
        
        # Apply dropout to decoder output during training
        if self.training:
            decoder_output = self.decoder_dropout(decoder_output)
        
        # Apply segmentation head to get final logits
        logits = self.model.segmentation_head(decoder_output)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        images, labels = batch
        logits = self(images)
        
        # Compute loss
        loss = self.criterion(logits, labels)
        
        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        
        # Store outputs for epoch-level metrics
        self.training_step_outputs.append({
            'loss': loss.detach(),
            'preds': preds.detach().cpu(),
            'labels': labels.detach().cpu()
        })
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        images, labels = batch
        logits = self(images)
        
        # Compute loss
        loss = self.criterion(logits, labels)
        
        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store outputs for epoch-level metrics
        self.validation_step_outputs.append({
            'loss': loss.detach(),
            'preds': preds.detach().cpu(),
            'labels': labels.detach().cpu()
        })
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        images, labels = batch
        logits = self(images)
        
        # Compute loss
        loss = self.criterion(logits, labels)
        
        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', accuracy, on_step=False, on_epoch=True)
        
        # Store outputs for epoch-level metrics
        self.test_step_outputs.append({
            'loss': loss.detach(),
            'preds': preds.detach().cpu(),
            'labels': labels.detach().cpu()
        })
        
        return loss
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch"""
        if len(self.training_step_outputs) == 0:
            return
            
        # Compute epoch-level metrics
        all_preds = torch.cat([x['preds'] for x in self.training_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.training_step_outputs])
        
        # Flatten for sklearn metrics
        all_preds_flat = all_preds.numpy().flatten()
        all_labels_flat = all_labels.numpy().flatten()
        
        # Compute F1 and IoU scores
        f1_macro = f1_score(all_labels_flat, all_preds_flat, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels_flat, all_preds_flat, average='weighted', zero_division=0)
        iou_macro = jaccard_score(all_labels_flat, all_preds_flat, average='macro', zero_division=0)
        
        self.log('train_f1_macro', f1_macro, on_epoch=True)
        self.log('train_f1_weighted', f1_weighted, on_epoch=True)
        self.log('train_iou_macro', iou_macro, on_epoch=True)
        
        # Clear outputs
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        if len(self.validation_step_outputs) == 0:
            return
            
        # Compute epoch-level metrics
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        
        # Flatten for sklearn metrics
        all_preds_flat = all_preds.numpy().flatten()
        all_labels_flat = all_labels.numpy().flatten()
        
        # Compute F1 and IoU scores
        f1_macro = f1_score(all_labels_flat, all_preds_flat, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels_flat, all_preds_flat, average='weighted', zero_division=0)
        iou_macro = jaccard_score(all_labels_flat, all_preds_flat, average='macro', zero_division=0)
        
        self.log('val_f1_macro', f1_macro, on_epoch=True)
        self.log('val_f1_weighted', f1_weighted, on_epoch=True)
        self.log('val_iou_macro', iou_macro, on_epoch=True)
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def on_test_epoch_end(self):
        """Called at the end of test epoch"""
        if len(self.test_step_outputs) == 0:
            return
            
        # Compute epoch-level metrics
        all_preds = torch.cat([x['preds'] for x in self.test_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs])
        
        # Flatten for sklearn metrics
        all_preds_flat = all_preds.numpy().flatten()
        all_labels_flat = all_labels.numpy().flatten()
        
        # Compute F1 and IoU scores
        f1_macro = f1_score(all_labels_flat, all_preds_flat, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels_flat, all_preds_flat, average='weighted', zero_division=0)
        iou_macro = jaccard_score(all_labels_flat, all_preds_flat, average='macro', zero_division=0)
        
        # Per-class metrics
        f1_per_class = f1_score(all_labels_flat, all_preds_flat, average=None, zero_division=0)
        iou_per_class = jaccard_score(all_labels_flat, all_preds_flat, average=None, zero_division=0)
        
        # Debug information
        print(f"DEBUG: self.num_classes = {self.num_classes}")
        print(f"DEBUG: f1_per_class.shape = {f1_per_class.shape}")
        print(f"DEBUG: unique labels in test data = {sorted(np.unique(all_labels_flat))}")
        print(f"DEBUG: unique preds in test data = {sorted(np.unique(all_preds_flat))}")
        print(f"DEBUG: max label = {all_labels_flat.max()}, max pred = {all_preds_flat.max()}")
        
        self.log('test_f1_macro', f1_macro, on_epoch=True)
        self.log('test_f1_weighted', f1_weighted, on_epoch=True)
        self.log('test_iou_macro', iou_macro, on_epoch=True)
        
        # Log per-class metrics (only for classes present in the data)
        for i in range(len(f1_per_class)):
            self.log(f'test_f1_class_{i}', f1_per_class[i], on_epoch=True)
            self.log(f'test_iou_class_{i}', iou_per_class[i], on_epoch=True)
        
        # Print detailed results
        print("\n=== Test Results ===")
        print(f"F1 Macro: {f1_macro:.4f}")
        print(f"F1 Weighted: {f1_weighted:.4f}")
        print(f"IoU Macro: {iou_macro:.4f}")
        print("\nPer-class F1 scores:")
        for i, f1 in enumerate(f1_per_class):
            print(f"  Class {i}: {f1:.4f}")
        print("\nPer-class IoU scores:")
        for i, iou in enumerate(iou_per_class):
            print(f"  Class {i}: {iou:.4f}")
        
        # Clear outputs
        self.test_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        optimizer = Adam(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        # More aggressive learning rate decay to reduce fluctuations
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.3,  # Reduce LR by 70% when triggered
            patience=3,  # Trigger after 3 epochs without improvement
            min_lr=1e-7,  # Minimum learning rate
            verbose=True,
            threshold=0.01,  # Minimum change to qualify as an improvement
            cooldown=2  # Wait 2 epochs before resuming normal operation
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
                "interval": "epoch",
            },
        }

def create_model(in_channels=12, num_classes=14, encoder_name="resnet34", learning_rate=1e-3, dropout_rate=0.3, weight_decay=1e-3):
    """Create U-Net model with random initialization (no pretrained weights)"""
    model = UNetSegmentation(
        in_channels=in_channels,
        num_classes=num_classes,
        encoder_name=encoder_name,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay
    )
    return model

if __name__ == "__main__":
    # Test model creation
    model = create_model(num_classes=14)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print("Model uses random initialization (no pretrained weights)")
    
    # Test forward pass
    x = torch.randn(2, 12, 128, 128)  # Batch of 2, 12 channels, 128x128
    with torch.no_grad():
        output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Model test passed!")