#!/usr/bin/env python3
"""
Inference script for U-Net model on validation set
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from PIL import Image
import rasterio
from datetime import datetime

from dataset import create_data_loaders
from model import UNetSegmentation
import pytorch_lightning as pl

def load_model_from_checkpoint(checkpoint_path, in_channels=12, num_classes=14):
    """Load trained model from checkpoint"""
    model = UNetSegmentation.load_from_checkpoint(
        checkpoint_path,
        in_channels=in_channels,
        num_classes=num_classes
    )
    model.eval()
    return model

def inference_on_validation_set(
    checkpoint_path,
    data_dir,
    n_images=5,
    batch_size=1,
    output_dir="./inference_results",
    label_level='L2'
):
    """Run inference on first N images from validation set"""
    
    print(f"=== Running Inference on Validation Set ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data directory: {data_dir}")
    print(f"Number of images: {n_images}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("\nLoading model...")
    num_classes = 14 if label_level == 'L2' else 7
    model = load_model_from_checkpoint(checkpoint_path, num_classes=num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    # Create validation data loader
    print("\nCreating validation data loader...")
    _, val_loader, _ = create_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=0,  # Set to 0 for inference
        label_level=label_level
    )
    
    # Class names for L2 (14 classes)
    l2_class_names = {
        0: 'Pa', 1: 'V1', 2: 'V2', 3: 'Wt', 4: 'Mg', 5: 'UA', 6: 'OB',
        7: 'Ft', 8: 'PR', 9: 'SP', 10: 'T1', 11: 'T1+', 12: 'OU', 13: 'Df'
    }
    
    # L1 class names (7 classes)
    l1_class_names = {
        0: 'Pasture', 1: 'Forest', 2: 'Agriculture', 3: 'Mining',
        4: 'Building', 5: 'Water body', 6: 'Other Uses'
    }
    
    class_names = l2_class_names if label_level == 'L2' else l1_class_names
    
    # Run inference
    print(f"\nRunning inference on first {n_images} validation images...")
    
    results = []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            if batch_idx >= n_images:
                break
                
            # Move to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Get predictions
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            
            # Convert to CPU and numpy
            image_np = images[0].cpu().numpy()  # (12, H, W)
            label_np = labels[0].cpu().numpy()  # (H, W)
            pred_np = predictions[0].cpu().numpy()  # (H, W)
            
            # Store results
            results.append({
                'image': image_np,
                'label': label_np,
                'prediction': pred_np,
                'batch_idx': batch_idx
            })
            
            # Calculate accuracy for this image
            accuracy = (pred_np == label_np).mean() * 100
            print(f"Image {batch_idx + 1}: Accuracy = {accuracy:.2f}%")
            
            # Create visualization
            create_visualization(
                image_np, label_np, pred_np, class_names,
                save_path=os.path.join(output_dir, f"inference_{batch_idx + 1:03d}.png"),
                title=f"Image {batch_idx + 1} - Accuracy: {accuracy:.2f}%"
            )
    
    print(f"\nInference completed! Results saved to: {output_dir}")
    return results

def create_visualization(image, label, prediction, class_names, save_path, title=""):
    """Create visualization of input, ground truth, and prediction"""
    
    # Create RGB composite from multispectral image (use bands 3, 2, 1 for RGB approximation)
    rgb_image = np.stack([image[3], image[2], image[1]], axis=-1)  # (H, W, 3)
    
    # Normalize RGB image for visualization
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
    rgb_image = np.clip(rgb_image, 0, 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot RGB image
    axes[0].imshow(rgb_image)
    axes[0].set_title('RGB Image (Bands 3,2,1)')
    axes[0].axis('off')
    
    # Plot ground truth
    im1 = axes[1].imshow(label, cmap='tab20', vmin=0, vmax=len(class_names)-1)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Plot prediction
    im2 = axes[2].imshow(prediction, cmap='tab20', vmin=0, vmax=len(class_names)-1)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im2, ax=axes, orientation='horizontal', 
                       fraction=0.046, pad=0.1, shrink=0.8)
    
    # Set colorbar ticks and labels
    ticks = list(range(len(class_names)))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([class_names[i] for i in ticks])
    cbar.ax.tick_params(labelsize=8)
    
    # Set main title
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Run inference on validation set')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to dataset directory')
    parser.add_argument('--n_images', type=int, default=5,
                        help='Number of validation images to process')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='Directory to save inference results')
    parser.add_argument('--label_level', type=str, default='L2', choices=['L1', 'L2'],
                        help='Label level: L1 (7 classes) or L2 (14 classes)')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file '{args.checkpoint_path}' not found!")
        return
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found!")
        return
    
    # Add timestamp to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"inference_{timestamp}")
    
    # Run inference
    results = inference_on_validation_set(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        n_images=args.n_images,
        batch_size=args.batch_size,
        output_dir=output_dir,
        label_level=args.label_level
    )
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Processed {len(results)} images")
    print(f"Results saved to: {output_dir}")
    
    # Calculate overall accuracy
    total_pixels = 0
    correct_pixels = 0
    for result in results:
        mask = (result['prediction'] == result['label'])
        correct_pixels += mask.sum()
        total_pixels += mask.size
    
    overall_accuracy = (correct_pixels / total_pixels) * 100
    print(f"Overall pixel accuracy: {overall_accuracy:.2f}%")

if __name__ == "__main__":
    main()