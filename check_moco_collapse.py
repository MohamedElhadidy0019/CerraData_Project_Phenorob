#!/usr/bin/env python3
"""
Script to check for representational collapse in MoCo training
Analyzes encoder outputs to detect if the model is learning diverse features
"""
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import segmentation_models_pytorch as smp

# ============================================================================
# CONFIGURATION - EDIT THESE PATHS
# ============================================================================

# Path to your MoCo checkpoint (the .ckpt file from PyTorch Lightning)
CHECKPOINT_PATH = "./checkpoints_data_splitted/moco_pretrain_resnet34_v2_more_aggressive_20251218_114641/last.ckpt"

# Path to your data directory
DATA_DIR = "/home/s52melba/CerraData_Project_Phenorob/cerradata_splitted"

# Number of samples to analyze
NUM_SAMPLES = 100

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# DATA STATISTICS (from CerraData-4MM)
# ============================================================================

MEAN = np.array([1331.2999603920011, 1422.618248839035, 1648.7418838236356, 1811.0396095371318,
                 2243.6360604171587, 2862.469356914663, 3158.7246770243464, 3253.5804747400075,
                 3464.1887187200564, 3463.5260019211623, 3635.662557047575, 2740.6395025025904]).reshape(12, 1, 1).astype(np.float32)

STDDEV = np.array([436.04697715189127, 484.32797096427566, 549.125419913045, 741.2668466992163,
                   788.8006282648606, 860.9668486457188, 963.2983618801512, 1000.2677835011111,
                   1087.111000434025, 1062.9960118331512, 1373.6088616321088, 1125.5168224477407]).reshape(12, 1, 1).astype(np.float32)


# ============================================================================
# ENCODER WRAPPER
# ============================================================================

class EncoderWrapper(nn.Module):
    """Wrapper to extract encoder from MoCo checkpoint"""
    def __init__(self, checkpoint_path):
        super().__init__()

        # Load checkpoint
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Create encoder
        base_unet = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=12,
            classes=14
        )
        self.encoder = base_unet.encoder

        # Load encoder weights from checkpoint
        # MoCo saves weights as 'backbone.xxx'
        encoder_state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            if key.startswith('backbone.'):
                new_key = key.replace('backbone.', '')
                encoder_state_dict[new_key] = value

        self.encoder.load_state_dict(encoder_state_dict)
        print("Encoder loaded successfully!")

    def forward(self, x):
        """Extract features from encoder"""
        features = self.encoder(x)
        h = features[-1]  # (B, 512, H', W')
        h = nn.functional.adaptive_avg_pool2d(h, 1)  # (B, 512, 1, 1)
        h = h.flatten(1)  # (B, 512)
        return h


# ============================================================================
# DATA LOADING
# ============================================================================

def load_image(img_path):
    """Load and normalize a single MSI image"""
    with rasterio.open(img_path) as src:
        image = src.read()  # (12, H, W)
        image = image.astype(np.float32)

    # Z-score normalization
    normalized = (image - MEAN) / (STDDEV + 1e-8)

    return torch.from_numpy(normalized).float()


def load_dataset_samples(data_dir, num_samples):
    """Load sample images from the dataset"""
    images_dir = Path(data_dir) / "train" / "images"

    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")

    image_files = list(images_dir.glob("*.tif"))
    image_files.sort()

    if len(image_files) == 0:
        raise ValueError(f"No images found in {images_dir}")

    # Sample random images
    num_samples = min(num_samples, len(image_files))
    indices = np.random.choice(len(image_files), num_samples, replace=False)

    print(f"Loading {num_samples} images from {images_dir}...")
    images = []
    for idx in indices:
        img = load_image(image_files[idx])
        images.append(img)

    # Stack into batch
    images = torch.stack(images)  # (N, 12, H, W)
    print(f"Loaded images shape: {images.shape}")

    return images


# ============================================================================
# COLLAPSE DETECTION
# ============================================================================

def compute_feature_statistics(features):
    """Compute statistics to detect collapse"""
    # features: (N, 512) tensor

    features_np = features.cpu().numpy()

    stats = {}

    # 1. Standard deviation per dimension (averaged across dimensions)
    std_per_dim = features_np.std(axis=0)  # (512,)
    stats['mean_std'] = std_per_dim.mean()
    stats['min_std'] = std_per_dim.min()
    stats['max_std'] = std_per_dim.max()

    # 2. Feature norms
    norms = np.linalg.norm(features_np, axis=1)  # (N,)
    stats['mean_norm'] = norms.mean()
    stats['std_norm'] = norms.std()

    # 3. Pairwise cosine similarities
    # Normalize features
    features_normalized = features_np / (np.linalg.norm(features_np, axis=1, keepdims=True) + 1e-8)

    # Compute similarity matrix
    similarity_matrix = features_normalized @ features_normalized.T  # (N, N)

    # Get upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(similarity_matrix), k=1).astype(bool)
    similarities = similarity_matrix[mask]

    stats['mean_similarity'] = similarities.mean()
    stats['std_similarity'] = similarities.std()
    stats['max_similarity'] = similarities.max()
    stats['min_similarity'] = similarities.min()

    # 4. Rank of feature matrix (effective dimensionality)
    U, S, Vt = np.linalg.svd(features_np, full_matrices=False)

    # Effective rank (number of singular values > threshold)
    threshold = 0.01 * S[0]  # 1% of largest singular value
    effective_rank = (S > threshold).sum()
    stats['effective_rank'] = effective_rank
    stats['rank_ratio'] = effective_rank / features_np.shape[1]  # ratio to max possible

    # 5. Explained variance by top components
    variance_explained = (S ** 2) / (S ** 2).sum()
    stats['var_explained_top1'] = variance_explained[0]
    stats['var_explained_top10'] = variance_explained[:10].sum()
    stats['var_explained_top50'] = variance_explained[:50].sum()

    return stats, features_np, similarity_matrix, S


def detect_collapse(stats):
    """Determine if collapse has occurred based on statistics"""

    collapse_indicators = []

    # Check 1: Low feature std (features not diverse)
    if stats['mean_std'] < 0.1:
        collapse_indicators.append(f"⚠️  COLLAPSE: Very low mean std ({stats['mean_std']:.4f} < 0.1)")
    elif stats['mean_std'] < 0.5:
        collapse_indicators.append(f"⚠️  WARNING: Low mean std ({stats['mean_std']:.4f} < 0.5)")
    else:
        collapse_indicators.append(f"✓ Good mean std: {stats['mean_std']:.4f}")

    # Check 2: High pairwise similarity (all samples similar)
    if stats['mean_similarity'] > 0.95:
        collapse_indicators.append(f"⚠️  COLLAPSE: Very high mean similarity ({stats['mean_similarity']:.4f} > 0.95)")
    elif stats['mean_similarity'] > 0.8:
        collapse_indicators.append(f"⚠️  WARNING: High mean similarity ({stats['mean_similarity']:.4f} > 0.8)")
    else:
        collapse_indicators.append(f"✓ Good mean similarity: {stats['mean_similarity']:.4f}")

    # Check 3: Low effective rank (low dimensionality)
    if stats['rank_ratio'] < 0.1:
        collapse_indicators.append(f"⚠️  COLLAPSE: Very low rank ratio ({stats['rank_ratio']:.4f} < 0.1)")
    elif stats['rank_ratio'] < 0.3:
        collapse_indicators.append(f"⚠️  WARNING: Low rank ratio ({stats['rank_ratio']:.4f} < 0.3)")
    else:
        collapse_indicators.append(f"✓ Good rank ratio: {stats['rank_ratio']:.4f}")

    # Check 4: High variance in top component (one dominant direction)
    if stats['var_explained_top1'] > 0.9:
        collapse_indicators.append(f"⚠️  COLLAPSE: Top component explains too much ({stats['var_explained_top1']:.4f} > 0.9)")
    elif stats['var_explained_top1'] > 0.5:
        collapse_indicators.append(f"⚠️  WARNING: Top component explains a lot ({stats['var_explained_top1']:.4f} > 0.5)")
    else:
        collapse_indicators.append(f"✓ Good variance distribution: top1={stats['var_explained_top1']:.4f}")

    # Overall verdict
    has_collapse = any("COLLAPSE" in indicator for indicator in collapse_indicators)
    has_warning = any("WARNING" in indicator for indicator in collapse_indicators)

    return collapse_indicators, has_collapse, has_warning


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_results(features_np, similarity_matrix, S, stats, save_dir="./collapse_analysis"):
    """Create visualizations of the collapse analysis"""

    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Similarity matrix heatmap
    ax = axes[0, 0]
    sns.heatmap(similarity_matrix[:50, :50], cmap='RdYlBu_r', center=0,
                vmin=-1, vmax=1, square=True, ax=ax, cbar_kws={'label': 'Cosine Similarity'})
    ax.set_title(f'Pairwise Cosine Similarity (first 50 samples)\nMean: {stats["mean_similarity"]:.3f}')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Sample Index')

    # 2. Singular value spectrum
    ax = axes[0, 1]
    ax.plot(S[:100], 'b-', linewidth=2)
    ax.axhline(y=0.01 * S[0], color='r', linestyle='--', label=f'1% threshold (rank={stats["effective_rank"]})')
    ax.set_xlabel('Component Index')
    ax.set_ylabel('Singular Value')
    ax.set_title('Singular Value Spectrum')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Cumulative explained variance
    ax = axes[1, 0]
    variance_explained = (S ** 2) / (S ** 2).sum()
    cumulative_var = np.cumsum(variance_explained)
    ax.plot(cumulative_var[:100], 'g-', linewidth=2)
    ax.axhline(y=0.9, color='r', linestyle='--', label='90% variance')
    ax.axhline(y=0.95, color='orange', linestyle='--', label='95% variance')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('Cumulative Variance Explained')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. PCA visualization (2D projection)
    ax = axes[1, 1]
    if features_np.shape[0] > 2:
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_np)
        ax.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6, s=50)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        ax.set_title('Features in 2D PCA Space')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'collapse_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.close()

    # Create histogram of similarities
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    mask = np.triu(np.ones_like(similarity_matrix), k=1).astype(bool)
    similarities = similarity_matrix[mask]
    ax.hist(similarities, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=stats['mean_similarity'], color='r', linestyle='--',
               linewidth=2, label=f'Mean: {stats["mean_similarity"]:.3f}')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Pairwise Cosine Similarities')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(save_dir, 'similarity_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Similarity distribution saved to: {save_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("MoCo Representational Collapse Detector")
    print("=" * 80)

    # Check if checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n❌ ERROR: Checkpoint not found at: {CHECKPOINT_PATH}")
        print("\nPlease update CHECKPOINT_PATH in the script to point to your .ckpt file")
        print("Example: ./checkpoints_data_splitted/moco_pretrain_resnet34_20241216_123456/last.ckpt")
        return

    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"\n❌ ERROR: Data directory not found: {DATA_DIR}")
        print("\nPlease update DATA_DIR in the script")
        return

    print(f"\nConfiguration:")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  Data dir: {DATA_DIR}")
    print(f"  Num samples: {NUM_SAMPLES}")
    print(f"  Device: {DEVICE}")

    # Load model
    print("\n" + "=" * 80)
    print("Loading Model")
    print("=" * 80)
    model = EncoderWrapper(CHECKPOINT_PATH)
    model = model.to(DEVICE)
    model.eval()

    # Load data
    print("\n" + "=" * 80)
    print("Loading Data")
    print("=" * 80)
    images = load_dataset_samples(DATA_DIR, NUM_SAMPLES)

    # Extract features
    print("\n" + "=" * 80)
    print("Extracting Features")
    print("=" * 80)
    all_features = []
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(DEVICE)
            features = model(batch)
            all_features.append(features.cpu())
            print(f"  Processed {min(i+batch_size, len(images))}/{len(images)} images")

    features = torch.cat(all_features, dim=0)  # (N, 512)
    print(f"Extracted features shape: {features.shape}")

    # Compute statistics
    print("\n" + "=" * 80)
    print("Computing Statistics")
    print("=" * 80)
    stats, features_np, similarity_matrix, S = compute_feature_statistics(features)

    print("\nFeature Statistics:")
    print(f"  Mean std across dimensions: {stats['mean_std']:.4f}")
    print(f"  Std range: [{stats['min_std']:.4f}, {stats['max_std']:.4f}]")
    print(f"  Mean feature norm: {stats['mean_norm']:.4f} ± {stats['std_norm']:.4f}")
    print(f"\nSimilarity Statistics:")
    print(f"  Mean pairwise similarity: {stats['mean_similarity']:.4f}")
    print(f"  Std of similarities: {stats['std_similarity']:.4f}")
    print(f"  Similarity range: [{stats['min_similarity']:.4f}, {stats['max_similarity']:.4f}]")
    print(f"\nDimensionality:")
    print(f"  Effective rank: {stats['effective_rank']}/512 ({stats['rank_ratio']:.1%})")
    print(f"  Variance explained by top 1 component: {stats['var_explained_top1']:.1%}")
    print(f"  Variance explained by top 10 components: {stats['var_explained_top10']:.1%}")
    print(f"  Variance explained by top 50 components: {stats['var_explained_top50']:.1%}")

    # Detect collapse
    print("\n" + "=" * 80)
    print("Collapse Detection")
    print("=" * 80)
    indicators, has_collapse, has_warning = detect_collapse(stats)

    for indicator in indicators:
        print(f"  {indicator}")

    print("\n" + "=" * 80)
    if has_collapse:
        print("❌ REPRESENTATIONAL COLLAPSE DETECTED!")
        print("=" * 80)
        print("\nThe model is NOT learning diverse features. Recommendations:")
        print("  1. Reduce learning rate (try 0.01 or 0.001)")
        print("  2. Increase temperature (try 0.5)")
        print("  3. Reduce memory bank size (try 4096)")
        print("  4. Check if warmup is working correctly")
        print("  5. Consider adding weight decay regularization")
    elif has_warning:
        print("⚠️  WARNING: Potential issues detected")
        print("=" * 80)
        print("\nThe model shows some concerning patterns. Monitor training closely.")
        print("Consider adjusting hyperparameters if loss plateaus.")
    else:
        print("✓ MODEL IS LEARNING WELL!")
        print("=" * 80)
        print("\nNo collapse detected. Features are diverse and well-distributed.")

    # Visualize
    print("\n" + "=" * 80)
    print("Creating Visualizations")
    print("=" * 80)
    visualize_results(features_np, similarity_matrix, S, stats)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
