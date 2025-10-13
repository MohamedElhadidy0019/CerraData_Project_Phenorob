#!/usr/bin/env python3
"""
Dataset classes for CerraData-4MM
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import rasterio
from pathlib import Path
from sklearn.model_selection import train_test_split
import json

class CerraDataset(Dataset):
    """Dataset class for CerraData-4MM"""
    
    def __init__(self, data_dir, split='train', label_level='L2', transform=None, 
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        """
        Args:
            data_dir: Path to dataset directory
            split: 'train', 'val', or 'test'
            label_level: 'L1' (7 classes) or 'L2' (14 classes)
            transform: Optional transforms
            train_ratio: Training split ratio (default 0.7)
            val_ratio: Validation split ratio (default 0.15)
            test_ratio: Test split ratio (default 0.15)
            random_state: Random seed for reproducible splits
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.label_level = label_level
        self.transform = transform
        
        # Find all image files
        self.images_dir = self.data_dir / "msi_images"
        # Set labels directory based on label level
        if label_level == 'L2':
            self.labels_dir = self.data_dir / "semantic_14c"
        else:  # L1
            self.labels_dir = self.data_dir / "semantic_7c"
        
        # Get all image files
        image_files = list(self.images_dir.glob("*.tif"))
        image_files.sort()  # Ensure consistent ordering
        
        # Create train/val/test splits
        train_files, temp_files = train_test_split(
            image_files, test_size=(val_ratio + test_ratio), 
            random_state=random_state
        )
        
        val_files, test_files = train_test_split(
            temp_files, test_size=test_ratio/(val_ratio + test_ratio),
            random_state=random_state
        )
        
        # Select files based on split
        if split == 'train':
            self.image_files = train_files
        elif split == 'val':
            self.image_files = val_files
        elif split == 'test':
            self.image_files = test_files
        else:
            raise ValueError(f"Invalid split: {split}")
        
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
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load multispectral image (12 channels)
        img_path = self.image_files[idx]
        with rasterio.open(img_path) as src:
            image = src.read()  # Shape: (12, H, W)
            image = image.astype(np.float32)
        
        # Load label
        # Extract the numeric ID from the image filename (e.g., "parrot_beak_ms_42312" -> "42312")
        img_id = img_path.stem.split('_')[-1]
        if self.label_level == 'L2':
            label_filename = f"parrot_beak_terraclass_classes_14c_{img_id}.tif"
        else:  # L1
            label_filename = f"parrot_beak_terraclass_classes_7c_{img_id}.tif"
        label_path = self.labels_dir / label_filename
        with rasterio.open(label_path) as src:
            label = src.read(1)  # Shape: (H, W)
            label = label.astype(np.int64)
        
        # Convert L2 to L1 if needed
        if self.label_level == 'L1':
            label_l1 = np.zeros_like(label)
            for l2_class, l1_class in self.l2_to_l1.items():
                label_l1[label == l2_class] = l1_class
            label = label_l1
        
        # Convert to torch tensors
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        
        # Normalize image (simple min-max normalization)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        if self.transform:
            # Apply transforms (need to handle both image and label)
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
            image, label = sample['image'], sample['label']
        
        return image, label

def create_data_loaders(data_dir, batch_size=16, num_workers=4, label_level='L2'):
    """Create train, validation, and test data loaders"""
    
    # Create datasets
    train_dataset = CerraDataset(data_dir, split='train', label_level=label_level)
    val_dataset = CerraDataset(data_dir, split='val', label_level=label_level)
    test_dataset = CerraDataset(data_dir, split='test', label_level=label_level)
    
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
    data_dir = "./data"
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
        print("Data directory not found. Run download_data.py first.")