#!/usr/bin/env python3
"""
Dataset classes for CerraData-4MM with physical splits
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import rasterio
from pathlib import Path
import random

class CerraDataset(Dataset):
    """Dataset class for physically split CerraData-4MM"""
    
    def __init__(self, data_dir, split='train', label_level='L2', transform=None, global_stats=None):
        """
        Args:
            data_dir: Path to physically split dataset directory
            split: 'train', 'val', or 'test'
            label_level: 'L1' (7 classes) or 'L2' (14 classes)
            transform: Optional transforms
            global_stats: Optional tuple of (global_min, global_max) for normalization.
                         If None, will compute from current split.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.label_level = label_level
        self.transform = transform

        # Set paths for physically split dataset
        self.split_dir = self.data_dir / split
        self.images_dir = self.split_dir / "images"

        if label_level == 'L2':
            self.labels_dir = self.split_dir / "labels_l2"
        else:  # L1
            self.labels_dir = self.split_dir / "labels_l1"

        # Check directories exist
        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise ValueError(f"Labels directory not found: {self.labels_dir}")

        # Get all image files
        self.image_files = list(self.images_dir.glob("*.tif"))
        self.image_files.sort()  # Ensure consistent ordering

        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {self.images_dir}")

        print(f"{split.upper()} split: {len(self.image_files)} samples")

        # L2 class mapping (14 classes)
        self.l2_classes = {
            'Pa': 0, 'V1': 1, 'V2': 2, 'Wt': 3, 'Mg': 4, 'UA': 5, 'OB': 6,
            'Ft': 7, 'PR': 8, 'SP': 9, 'T1': 10, 'T1+': 11, 'OU': 12, 'Df': 13
        }

        # L1 class mapping (7 classes)
        self.l1_classes = {
            'Pasture': 0, 'Forest': 1, 'Agriculture': 2, 'Mining': 3,
            'Building': 4, 'Water body': 5, 'Other Uses': 6
        }

        # L2 to L1 mapping
        self.l2_to_l1 = {
            0: 0,  # Pa -> Pasture
            1: 1, 2: 1, 7: 1,  # V1, V2, Ft -> Forest
            8: 2, 9: 2, 10: 2, 11: 2,  # PR, SP, T1, T1+ -> Agriculture
            4: 3,  # Mg -> Mining
            5: 4, 6: 4,  # UA, OB -> Building
            3: 5,  # Wt -> Water body
            12: 6, 13: 6  # OU, Df -> Other Uses
        }

        self.num_classes = 14 if label_level == 'L2' else 7

        # Set or compute global statistics for normalization
        if global_stats is not None:
            self.global_min, self.global_max = global_stats
            print(f"Using provided global statistics")
        else:
            print(f"Computing global per-channel statistics from {split} split...")
            self.global_min, self.global_max = self._compute_global_stats()
            print(f"Global min per channel: {self.global_min}")
            print(f"Global max per channel: {self.global_max}")

    def _compute_global_stats(self):
        """Compute global min/max per channel across all images in this split"""
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
        
        # Load corresponding label
        img_id = img_path.stem.split('_')[-1]
        
        if self.label_level == 'L2':
            label_filename = f"parrot_beak_terraclass_classes_14c_{img_id}.tif"
        else:  # L1
            label_filename = f"mask_parrot_beak_7classes_{img_id}.tif"
            
        label_path = self.labels_dir / label_filename
        
        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")
            
        with rasterio.open(label_path) as src:
            label = src.read(1)  # Shape: (H, W)
            label = label.astype(np.int64)
        
        # Convert L2 to L1 if needed
        if self.label_level == 'L1':
            label_l1 = np.zeros_like(label)
            for l2_class, l1_class in self.l2_to_l1.items():
                label_l1[label == l2_class] = l1_class
            label = label_l1

        # Per-channel normalization using global statistics
        normalized_image = np.zeros_like(image)
        for ch in range(12):
            ch_min = self.global_min[ch]
            ch_max = self.global_max[ch]
            normalized_image[ch] = (image[ch] - ch_min) / (ch_max - ch_min + 1e-8)

        # Convert to torch tensors
        image = torch.from_numpy(normalized_image)
        label = torch.from_numpy(label)

        if self.transform:
            # Apply transforms (need to handle both image and label)
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
            image, label = sample['image'], sample['label']

        return image, label

def create_data_loaders(data_dir, batch_size=16, num_workers=4, label_level='L2', data_percentage=100):
    """Create train, validation, and test data loaders from physically split dataset

    Global statistics are computed from the training set and shared across all splits
    to ensure consistent normalization.
    """

    # Create training dataset first (computes global statistics)
    print("\n=== Creating Training Dataset ===")
    train_dataset = CerraDataset(data_dir, split='train', label_level=label_level)

    # Extract global statistics from training set
    global_stats = (train_dataset.global_min, train_dataset.global_max)

    # Create val and test datasets using the same statistics
    print("\n=== Creating Validation Dataset ===")
    val_dataset = CerraDataset(data_dir, split='val', label_level=label_level, global_stats=global_stats)
    print("\n=== Creating Test Dataset ===")
    test_dataset = CerraDataset(data_dir, split='test', label_level=label_level, global_stats=global_stats)
    
    # Apply data percentage reduction if specified
    # For training: use percentage of train and val, but ALWAYS use full test set
    if data_percentage < 100:
        random.seed(42)  # For reproducibility
        
        # Calculate subset sizes (only for train and val)
        train_size = int(len(train_dataset) * data_percentage / 100)
        val_size = int(len(val_dataset) * data_percentage / 100)
        # test_size = len(test_dataset)  # Always use full test set
        
        # Create random indices for train and val only
        train_indices = random.sample(range(len(train_dataset)), train_size)
        val_indices = random.sample(range(len(val_dataset)), val_size)
        
        # Create subset datasets
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        # test_dataset remains unchanged (full test set)
        
        print(f"Using {data_percentage}% of train/val data:")
        print(f"  Train samples: {len(train_dataset)} ({data_percentage}% of original)")
        print(f"  Val samples: {len(val_dataset)} ({data_percentage}% of original)")
        print(f"  Test samples: {len(test_dataset)} (100% - full test set)")
    else:
        print(f"Using full dataset (100%)")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test dataset loading
    data_dir = "./data_split"
    if os.path.exists(data_dir):
        train_loader, val_loader, test_loader = create_data_loaders(data_dir, batch_size=2)
        
        # Test loading a batch
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"Batch {batch_idx}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
            print(f"  Unique labels: {torch.unique(labels)}")
            if batch_idx >= 2:  # Only test a few batches
                break
    else:
        print("Split dataset directory not found. Run create_physical_splits.py first.")