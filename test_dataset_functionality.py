#!/usr/bin/env python3
"""
Dummy script to test dataset loading, validation, and testing functionality
"""
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import CerraDataset
from model import create_model
import pytorch_lightning as pl


def test_dataset_splits(data_dir, label_level='L1', batch_size=4):
    """Test dataset loading and splitting"""
    print(f"\n=== Testing Dataset Splits ===")
    print(f"Data directory: {data_dir}")
    print(f"Label level: {label_level}")
    
    try:
        # Create datasets
        train_dataset = CerraDataset(data_dir, split='train', label_level=label_level)
        val_dataset = CerraDataset(data_dir, split='val', label_level=label_level)
        test_dataset = CerraDataset(data_dir, split='test', label_level=label_level)
        
        print(f"✓ Train dataset: {len(train_dataset)} samples")
        print(f"✓ Val dataset: {len(val_dataset)} samples")
        print(f"✓ Test dataset: {len(test_dataset)} samples")
        print(f"✓ Total samples: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
        
        # Test data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        print(f"✓ Train batches: {len(train_loader)}")
        print(f"✓ Val batches: {len(val_loader)}")
        print(f"✓ Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader, train_dataset.num_classes
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return None, None, None, None


def test_single_batch(data_loader, split_name):
    """Test loading a single batch from data loader"""
    print(f"\n=== Testing {split_name} Batch ===")
    
    try:
        batch = next(iter(data_loader))
        images = batch['image']
        labels = batch['label']
        
        print(f"✓ Batch loaded successfully")
        print(f"✓ Image shape: {images.shape}")
        print(f"✓ Label shape: {labels.shape}")
        print(f"✓ Image dtype: {images.dtype}")
        print(f"✓ Label dtype: {labels.dtype}")
        print(f"✓ Image range: [{images.min().item():.3f}, {images.max().item():.3f}]")
        print(f"✓ Label range: [{labels.min().item()}, {labels.max().item()}]")
        print(f"✓ Unique labels: {sorted(torch.unique(labels).tolist())}")
        
        return batch
        
    except Exception as e:
        print(f"✗ Batch loading failed: {e}")
        return None


def test_model_creation_and_inference(num_classes):
    """Test model creation and forward pass"""
    print(f"\n=== Testing Model Creation and Inference ===")
    
    try:
        # Create model
        model = create_model(
            in_channels=12,
            num_classes=num_classes,
            encoder_name="resnet34",
            learning_rate=1e-3
        )
        print(f"✓ Model created with {num_classes} classes")
        print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 12, 128, 128)  # Batch of 2
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✓ Forward pass successful")
            print(f"✓ Input shape: {dummy_input.shape}")
            print(f"✓ Output shape: {output.shape}")
            print(f"✓ Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        return model
        
    except Exception as e:
        print(f"✗ Model creation/inference failed: {e}")
        return None


def test_validation_step(model, val_loader):
    """Test validation step functionality"""
    print(f"\n=== Testing Validation Step ===")
    
    try:
        model.eval()
        batch = next(iter(val_loader))
        
        # Test validation step
        with torch.no_grad():
            val_loss = model.validation_step(batch, 0)
            print(f"✓ Validation step successful")
            print(f"✓ Validation loss: {val_loss}")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation step failed: {e}")
        return False


def test_model_training_step(model, train_loader):
    """Test training step functionality"""
    print(f"\n=== Testing Training Step ===")
    
    try:
        model.train()
        batch = next(iter(train_loader))
        
        # Test training step
        train_loss = model.training_step(batch, 0)
        print(f"✓ Training step successful")
        print(f"✓ Training loss: {train_loss}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        return False


def test_full_validation_epoch(model, val_loader):
    """Test running validation on multiple batches"""
    print(f"\n=== Testing Full Validation Epoch ===")
    
    try:
        model.eval()
        total_loss = 0
        num_batches = min(5, len(val_loader))  # Test first 5 batches
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= num_batches:
                    break
                val_loss = model.validation_step(batch, i)
                total_loss += val_loss.item()
                print(f"  Batch {i+1}/{num_batches}: loss = {val_loss:.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"✓ Validation epoch test successful")
        print(f"✓ Average loss over {num_batches} batches: {avg_loss:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation epoch test failed: {e}")
        return False


def test_lightning_trainer(model, val_loader, test_loader):
    """Test PyTorch Lightning trainer validation functionality"""
    print(f"\n=== Testing PyTorch Lightning Trainer ===")
    
    try:
        # Create a simple trainer for testing
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator='cpu',
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False
        )
        
        # Test validation
        print("Testing trainer.validate()...")
        val_results = trainer.validate(model, val_loader, verbose=False)
        print(f"✓ Trainer validation successful")
        
        # Test testing (just a few batches to avoid long runtime)
        print("Testing trainer.test() on a subset...")
        test_subset = torch.utils.data.Subset(test_loader.dataset, range(0, min(20, len(test_loader.dataset))))
        test_subset_loader = DataLoader(test_subset, batch_size=4, shuffle=False)
        
        test_results = trainer.test(model, test_subset_loader, verbose=False)
        print(f"✓ Trainer testing successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Lightning trainer test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test dataset and model functionality')
    parser.add_argument('--data_dir', required=True, help='Path to dataset directory')
    parser.add_argument('--label_level', default='L1', choices=['L1', 'L2'], help='Label level (L1 or L2)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for testing')
    parser.add_argument('--skip_trainer', action='store_true', help='Skip PyTorch Lightning trainer tests')
    
    args = parser.parse_args()
    
    print("="*60)
    print("DATASET AND MODEL FUNCTIONALITY TEST")
    print("="*60)
    
    # Test dataset loading
    train_loader, val_loader, test_loader, num_classes = test_dataset_splits(
        args.data_dir, args.label_level, args.batch_size
    )
    
    if train_loader is None:
        print("\n✗ Dataset loading failed. Stopping tests.")
        return
    
    print(f"\n✓ Using {num_classes} classes for {args.label_level}")
    
    # Test individual batches
    train_batch = test_single_batch(train_loader, "Training")
    val_batch = test_single_batch(val_loader, "Validation")
    test_batch = test_single_batch(test_loader, "Test")
    
    if train_batch is None or val_batch is None or test_batch is None:
        print("\n✗ Batch loading failed. Stopping tests.")
        return
    
    # Test model creation and inference
    model = test_model_creation_and_inference(num_classes)
    if model is None:
        print("\n✗ Model creation failed. Stopping tests.")
        return
    
    # Test model steps
    test_model_training_step(model, train_loader)
    test_validation_step(model, val_loader)
    test_full_validation_epoch(model, val_loader)
    
    # Test PyTorch Lightning trainer (optional)
    if not args.skip_trainer:
        test_lightning_trainer(model, val_loader, test_loader)
    else:
        print("\n--- Skipping PyTorch Lightning trainer tests ---")
    
    print("\n" + "="*60)
    print("TEST SUMMARY COMPLETED")
    print("="*60)
    print("If all tests show ✓, your dataset and model functionality is working correctly!")


if __name__ == "__main__":
    main()