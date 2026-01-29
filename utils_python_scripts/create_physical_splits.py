#!/usr/bin/env python3
"""
Create physical train/val/test splits from CerraData dataset
"""
import os
import shutil
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split

def create_physical_splits(source_data_dir, target_data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Create physical train/val/test splits with same logic as original dataset
    
    Args:
        source_data_dir: Path to original CerraData dataset
        target_data_dir: Path where split dataset will be created
        train_ratio: Training split ratio (default 0.7)
        val_ratio: Validation split ratio (default 0.15)
        test_ratio: Test split ratio (default 0.15)
        random_state: Random seed for reproducible splits
    """
    
    print("=== Creating Physical Dataset Splits ===")
    print(f"Source: {source_data_dir}")
    print(f"Target: {target_data_dir}")
    print(f"Split ratios: {train_ratio:.1f}/{val_ratio:.1f}/{test_ratio:.1f}")
    print(f"Random state: {random_state}")
    
    source_path = Path(source_data_dir)
    target_path = Path(target_data_dir)
    
    # Check source directories exist
    images_dir = source_path / "msi_images"
    l1_labels_dir = source_path / "semantic_7c"
    l2_labels_dir = source_path / "semantic_14c"
    
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    if not l1_labels_dir.exists():
        raise ValueError(f"L1 labels directory not found: {l1_labels_dir}")
    if not l2_labels_dir.exists():
        raise ValueError(f"L2 labels directory not found: {l2_labels_dir}")
    
    # Get all image files (same logic as original dataset)
    image_files = list(images_dir.glob("*.tif"))
    image_files.sort()  # Ensure consistent ordering
    
    print(f"Found {len(image_files)} image files")
    
    # Create train/val/test splits (SAME LOGIC AS ORIGINAL)
    train_files, temp_files = train_test_split(
        image_files, test_size=(val_ratio + test_ratio), 
        random_state=random_state
    )
    
    val_files, test_files = train_test_split(
        temp_files, test_size=test_ratio/(val_ratio + test_ratio),
        random_state=random_state
    )
    
    print(f"Train files: {len(train_files)}")
    print(f"Val files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")
    
    # Create target directory structure
    target_path.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        (target_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (target_path / split / 'labels_l1').mkdir(parents=True, exist_ok=True)
        (target_path / split / 'labels_l2').mkdir(parents=True, exist_ok=True)
    
    # Copy files to appropriate splits
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    for split_name, file_list in splits.items():
        print(f"\nCopying {split_name} files...")
        
        for img_file in file_list:
            # Extract the numeric ID from filename (e.g., "parrot_beak_ms_42312" -> "42312")
            img_id = img_file.stem.split('_')[-1]
            
            # Copy image
            target_img_path = target_path / split_name / 'images' / img_file.name
            shutil.copy2(img_file, target_img_path)
            
            # Copy L1 label
            l1_label_filename = f"mask_parrot_beak_7classes_{img_id}.tif"
            l1_source_path = l1_labels_dir / l1_label_filename
            l1_target_path = target_path / split_name / 'labels_l1' / l1_label_filename
            
            if l1_source_path.exists():
                shutil.copy2(l1_source_path, l1_target_path)
            else:
                print(f"Warning: L1 label not found: {l1_source_path}")
            
            # Copy L2 label
            l2_label_filename = f"parrot_beak_terraclass_classes_14c_{img_id}.tif"
            l2_source_path = l2_labels_dir / l2_label_filename
            l2_target_path = target_path / split_name / 'labels_l2' / l2_label_filename
            
            if l2_source_path.exists():
                shutil.copy2(l2_source_path, l2_target_path)
            else:
                print(f"Warning: L2 label not found: {l2_source_path}")
    
    print(f"\n=== Physical splits created successfully! ===")
    print(f"Dataset saved to: {target_path}")
    
    # Print summary
    for split in ['train', 'val', 'test']:
        img_count = len(list((target_path / split / 'images').glob("*.tif")))
        l1_count = len(list((target_path / split / 'labels_l1').glob("*.tif")))
        l2_count = len(list((target_path / split / 'labels_l2').glob("*.tif")))
        print(f"{split.upper()}: {img_count} images, {l1_count} L1 labels, {l2_count} L2 labels")

def main():
    parser = argparse.ArgumentParser(description='Create physical train/val/test splits from CerraData')
    parser.add_argument('--source_dir', type=str, required=True,
                        help='Path to original CerraData dataset directory')
    parser.add_argument('--target_dir', type=str, required=True,
                        help='Path where split dataset will be created')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training split ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation split ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test split ratio (default: 0.15)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducible splits (default: 42)')
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        print("Error: Train, val, and test ratios must sum to 1.0")
        return
    
    create_physical_splits(
        source_data_dir=args.source_dir,
        target_data_dir=args.target_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state
    )

if __name__ == "__main__":
    main()