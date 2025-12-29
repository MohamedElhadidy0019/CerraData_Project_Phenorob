# CerraData Project - Training Procedures Overview

## Project Summary

Semantic segmentation project for multispectral satellite imagery (Sentinel-2) using the CerraData-4MM dataset. Compares three learning approaches: supervised baseline, hierarchical pretraining, and self-supervised learning.

**Architecture**: U-Net with ResNet34 encoder (segmentation-models-pytorch)
**Input**: 12-channel Sentinel-2 multispectral images (128x128 pixels)
**Dataset**: CerraData-4MM

---

## Label Hierarchy

### L1 (Coarse) - 7 Classes
- Pasture
- Forest
- Agriculture
- Mining
- Building
- Water body
- Other Uses

### L2 (Fine-grained) - 14 Classes
Pa, V1, V2, Wt, Mg, UA, OB, Ft, PR, SP, T1, T1+, OU, Df

---

## Training Scripts

### 1. `train_baseline.py`
**Purpose**: Train U-Net from scratch with random initialization

**Features**:
- Random weight initialization (no pretrained weights)
- Supports both L1 and L2 via `--label_level` flag
- Standard supervised learning
- Data percentage control for limited-data experiments

**Key Parameters**:
- Learning rate: 1e-3
- Optimizer: Adam with ReduceLROnPlateau
- Early stopping patience: 20 epochs (configurable via `--patience`)
- Batch size: 100
- Data percentage: Configurable via `--data_percentage` (0.1-100)

**Used by**:
- `run_l1_pretrain.sh` - L1 baseline (7 classes, 100 epochs, 100% data)
- `run_l2_baseline.sh` - L2 baseline (14 classes, 200 epochs, 5% data)

---

### 2. `train_l2_finetune.py`
**Purpose**: Hierarchical transfer learning from L1 to L2

**Approach**:
1. Load pretrained L1 model checkpoint (7 classes)
2. Transfer encoder + decoder weights to new L2 model
3. Randomly initialize segmentation head for 14 classes
4. Fine-tune entire model on L2 labels

**Key Parameters**:
- Learning rate: 1e-4 (lower than baseline)
- Early stopping patience: 20 epochs (configurable via `--patience`)
- Data percentage: Configurable via `--data_percentage`
- Inherits other settings from baseline

**Used by**:
- `run_l2_finetune.sh` - 5% data, 100 epochs

**Research Question**: Does learning coarse labels first help fine-grained segmentation?

---

### 3. `train_simclr_lightly.py`
**Purpose**: Self-supervised pretraining using SimCLR contrastive learning

**Approach**:
- Framework: SimCLR (Simple Framework for Contrastive Learning)
- Data: Unlabeled training images only (no labels)
- Method: Contrastive learning - maximize agreement between augmented views
- Loss: NT-Xent (Normalized Temperature-scaled Cross Entropy)
- Output: Pretrained encoder weights (`encoder_final.pth`)

**Augmentation Pipeline** (MultispectralSimCLRTransform):
- **Geometric**: Random crop (0.2-1.0), horizontal/vertical flip, rotation
- **Spectral** (MSI-specific):
  - Band dropout: Randomly zero out up to 3 spectral bands
  - Intensity jitter: Per-band brightness/contrast variation
  - Gaussian noise: Simulate sensor noise
  - Solarization: Invert pixels above threshold

**Key Parameters**:
- Batch size: 512
- Epochs: 1000
- Learning rate: 2e-3 with warmup + cosine annealing
- Temperature: 0.3
- Projection dim: 128

**Used by**:
- `run_simclr_pretrain.sh`

---

### 4. `train_moco_lightly.py`
**Purpose**: Self-supervised pretraining using MoCo v2

**Approach**:
- Framework: MoCo v2 (Momentum Contrast version 2)
- Method:
  - Query encoder (updated via backprop)
  - Momentum encoder (EMA updates)
  - Memory bank: Queue of 65,536 negative samples
- Loss: NT-Xent with memory bank
- Output: Pretrained encoder weights

**Key Differences from SimCLR**:
- Uses momentum encoder instead of same-batch negatives
- Larger effective batch size via memory bank
- SGD optimizer instead of Adam
- Lower temperature to prevent collapse

**Key Parameters**:
- Batch size: 256
- Epochs: 1000
- Learning rate: 0.05
- Temperature: 0.07
- Momentum: Cosine schedule 0.996 → 1.0
- Memory bank: 65,536 samples
- Optimizer: SGD with momentum 0.9

**Used by**:
- `run_moco_pretrain.sh`

---

### 5. `train_l2_from_simclr.py`
**Purpose**: Fine-tune L2 segmentation using self-supervised pretrained encoder

**Approach**:
1. Create new L2 U-Net model (14 classes)
2. Load pretrained encoder weights from SimCLR/MoCo
3. Decoder randomly initialized
4. Fine-tune on labeled L2 data

**Key Parameters**:
- Batch size: 100
- Learning rate: 1e-4
- Early stopping patience: 40 epochs (configurable via `--patience`)
- Data percentage: Configurable via `--data_percentage`
- Typically used with limited labeled data (0.1%-10%)

**Used by**:
- `run_l2_from_self_supervision.sh`

**Research Question**: Does self-supervised pretraining help in low-data scenarios?

---

## Training Pipelines

### Pipeline 1: Supervised Baseline
```
Random Init → L2 Training (train_baseline.py)
```
- Direct training on L2 labels
- Baseline for comparison

### Pipeline 2: Hierarchical Supervised Learning
```
Random Init → L1 Training (train_baseline.py)
           → L2 Fine-tuning (train_l2_finetune.py)
```
- Pretraining on coarse labels (L1)
- Transfer to fine-grained labels (L2)
- Tests hierarchical curriculum learning

### Pipeline 3: Self-Supervised → Supervised
```
Unlabeled Data → SimCLR/MoCo Pretraining
               → L2 Fine-tuning (train_l2_from_simclr.py)
```
- Learn representations from unlabeled data
- Fine-tune on limited labeled data
- Two variants: SimCLR vs MoCo

---

## Shell Scripts Reference

### Pretraining Scripts
- `run_l1_pretrain.sh` → `train_baseline.py` (L1, 100% data)
- `run_simclr_pretrain.sh` → `train_simclr_lightly.py`
- `run_moco_pretrain.sh` → `train_moco_lightly.py`

### L2 Training Scripts (Single Runs)
- `run_l2_baseline.sh` → `train_baseline.py` (L2, 5% data)
- `run_l2_finetune.sh` → `train_l2_finetune.py` (from L1)
- `run_l2_from_self_supervision.sh` → `train_l2_from_simclr.py`

### L2 Training Scripts (Data Scaling Experiments)
**NEW**: These scripts loop through multiple data percentages automatically:

- `run_l2_baseline_loop.sh` → Baseline across all percentages
- `run_l2_finetune_loop.sh` → L1→L2 fine-tuning across all percentages
- `run_l2_from_supervision_loop.sh` → Self-supervised→L2 across all percentages

**Tested Percentages**: 0.1%, 0.5%, 1%, 2.5%, 5%, 10%, 25%, 50%, 100%

**Key Features**:
- Automatically runs experiments for all data percentages
- Experiment names include percentage (e.g., `l2_baseline_14classes_0_1percent`)
- Configurable patience parameter at top of each script
- Logs completion status for each percentage

**Before Running Loop Scripts**:
- Update checkpoint paths in fine-tuning scripts (L1 checkpoint, SimCLR/MoCo encoder)
- Adjust `PATIENCE` variable if needed (default: 20 for baseline/finetune, 40 for self-supervised)

### Utility Scripts
- `run_inference.sh` - Run inference on test set
- `run_test_model.sh` - Test trained model
- `run_dataset_test.sh` - Validate dataset loading
- `run_create_splits.sh` - Create train/val/test splits
- `run_debug.sh` - Debug mode training
- `run_examples.sh` - Example usage

---

## Experimental Variables

1. **Data Percentage**: 0.1%, 0.5%, 1%, 2.5%, 5%, 10%, 25%, 50%, 100% of training data
   - Train and validation sets reduced by percentage
   - Test set always kept at 100% for fair comparison
   - Logarithmic scale to capture low-data regime behavior
2. **Label Granularity**: L1 (7 classes) vs L2 (14 classes)
3. **Initialization Strategy**:
   - Random (baseline)
   - L1-pretrained (hierarchical)
   - Self-supervised pretrained (SimCLR/MoCo)
4. **Self-Supervised Method**: SimCLR vs MoCo v2
5. **Augmentation Strength**: Regular vs Strong transforms
6. **Early Stopping Patience**: 20 epochs (baseline/hierarchical) or 40 epochs (self-supervised)

---

## Research Questions

1. Does hierarchical pretraining (L1→L2) improve fine-grained segmentation?
2. Can self-supervised learning help with limited labeled data?
3. Which initialization strategy works best for multispectral satellite imagery?
4. How do SimCLR and MoCo compare for this domain?
5. **NEW**: How does performance scale with data percentage across different pretraining strategies?
   - At what data percentage do benefits of pretraining diminish?
   - Which method shows the strongest gains in very low data regimes (0.1%-1%)?

---

## Common Configurations

### Supervised Training (Baseline & Fine-tuning)
- Optimizer: Adam
- LR Scheduler: ReduceLROnPlateau
- Early Stopping: Yes (patience 20-40 epochs)
- Batch Size: 100

### Self-Supervised Training
- **SimCLR**:
  - Optimizer: Adam
  - LR Scheduler: Warmup + Cosine Annealing
  - Batch Size: 512
  - Temperature: 0.3

- **MoCo**:
  - Optimizer: SGD
  - Batch Size: 256
  - Temperature: 0.07
  - Memory Bank: 65,536

---

## Output Locations

- **Models**: Saved to directories specified in run scripts
- **Logs**: Training logs and metrics
- **Checkpoints**: Best model checkpoints saved based on validation performance
- **Pretrained Encoders**: `encoder_final.pth` from SimCLR/MoCo

---

## Data Scaling Experiments

The project includes automated data scaling experiments to compare how different pretraining strategies perform across varying amounts of labeled data.

### Experiment Setup

**Three Approaches Compared**:
1. **Baseline**: Random init → L2 training
2. **Hierarchical**: L1 pretrain (100% data) → L2 finetune
3. **Self-Supervised**: SimCLR/MoCo pretrain (unlabeled) → L2 finetune

**Data Percentages Tested**: 0.1, 0.5, 1, 2.5, 5, 10, 25, 50, 100

### Expected Results

- **Very Low Data (0.1-1%)**: Self-supervised and hierarchical should significantly outperform baseline
- **Low Data (2.5-10%)**: Gap narrows but pretraining still beneficial
- **High Data (25-100%)**: All methods converge, pretraining advantage diminishes

### Running Data Scaling Experiments

```bash
# 1. First, ensure you have pretrained checkpoints
# - L1 checkpoint from run_l1_pretrain.sh
# - SimCLR/MoCo encoder from run_simclr_pretrain.sh or run_moco_pretrain.sh

# 2. Update checkpoint paths in the loop scripts

# 3. Run all three experiments
sbatch run_l2_baseline_loop.sh
sbatch run_l2_finetune_loop.sh
sbatch run_l2_from_supervision_loop.sh

# 4. Results will be saved with percentage in experiment name
# - checkpoints_data_splitted/l2_baseline_14classes_0_1percent_TIMESTAMP/
# - checkpoints_data_splitted/l2_finetune_14classes_from_l1_0_1percent_TIMESTAMP/
# - checkpoints_data_splitted/l2_from_simclr_0_1percent_TIMESTAMP/
```

### Analysis

After running experiments:
- Compare validation metrics (F1, IoU) across percentages
- Plot learning curves: performance vs data percentage
- Identify optimal pretraining strategy for different data regimes
- Statistical significance testing across multiple runs

---

## Notes

- All experiments use the same U-Net + ResNet34 architecture for fair comparison
- Data scaling experiments use logarithmic percentages to capture low-data regime
- Test set always kept at 100% for consistent evaluation
- Multispectral-specific augmentations designed for 12-channel Sentinel-2 data
- Project addresses limited labeled data problem in satellite imagery segmentation
- Random seed (42) fixed for reproducibility across experiments
