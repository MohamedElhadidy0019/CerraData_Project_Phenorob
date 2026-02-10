# CerraData Project - Multimodal Crop Classification

Deep learning framework for hierarchical crop classification using multimodal satellite data (Sentinel-1 SAR + Sentinel-2 MSI) from the CerraData dataset.

## 📦 Installation

### Requirements

Install dependencies using the provided requirements file:

```bash
pip install -r requirements.txt
```

**⚠️ Important Note:** Some packages may require additional system-level dependencies and manual configuration:
- **GDAL/osgeo**: Geospatial data abstraction library - may need system GDAL installation first
- **tifffile**: GeoTIFF file handling - ensure compatible with your numpy version
- **scipy**: Scientific computing - may have specific build requirements

If you encounter issues with these packages, you may need to:
1. Install system-level GDAL: `sudo apt-get install gdal-bin libgdal-dev` (Linux) or use conda
2. Install packages individually with specific versions
3. Use conda for easier dependency management: `conda install gdal tifffile scipy`

---

## 📊 Dataset Preparation

**IMPORTANT**: Before training, you must split the dataset into train/val/test sets.

### Split Dataset

Use the provided script to split your raw CerraData into train (70%), validation (15%), and test (15%) sets:

```bash
python CerraData-4MM/split_dataset.py
```

**Note**: Update the `input_folder` and `output_folder` paths in the script to match your directory structure.

The script uses `splitfolders` with a fixed seed (42) for reproducibility.

---

## 📁 Dataset Loaders

### `dataset_loader_official/` Directory

This directory contains the **official dataloaders** for CerraData multimodal satellite data.

#### **`dataset_loader.py`** - L2 Dataset (14 Classes)
Used for training on the fine-grained L2 classification level (14 crop classes).

**Available Classes:**
- **`MMDataset`**: 14 channels (12 MSI + 2 SAR) - Multimodal fusion
- **`MSIDataset`**: 12 channels - Sentinel-2 optical data only
- **`SARDataset`**: 2 channels - Sentinel-1 radar data only

**Supported Normalization**: `"none"`, `"0to1"`, `"1to1"`, `"z_score"`

#### **`dataset_loader_7.py`** - L1 Dataset (7 Classes)
Used for training on the coarse L1 classification level (7 crop classes).

**Available Classes:**
- **`MMDataset`**: 14 channels (12 MSI + 2 SAR) - Multimodal fusion
- **`MSIDataset`**: 12 channels - Sentinel-2 optical data only
- **`SARDataset`**: 2 channels - Sentinel-1 radar data only

**Supported Normalization**: `"none"`, `"0to1"`, `"1to1"`, `"z_score"`

**Use Case**: Pretraining on L1 before fine-tuning on L2

#### Dataset Class Summary

| Dataset Class | File | Channels | Modalities | Use Case |
|--------------|------|----------|------------|----------|
| `MMDataset` | `dataset_loader.py` | 14 | MSI (12) + SAR (2) | L2 multimodal classification |
| `MSIDataset` | `dataset_loader.py` | 12 | MSI only | L2 optical-only classification |
| `SARDataset` | `dataset_loader.py` | 2 | SAR only | L2 radar-only classification |
| `MMDataset` | `dataset_loader_7.py` | 14 | MSI (12) + SAR (2) | L1 multimodal pretraining |
| `MSIDataset` | `dataset_loader_7.py` | 12 | MSI only | L1 optical-only pretraining |
| `SARDataset` | `dataset_loader_7.py` | 2 | SAR only | L1 radar-only pretraining |

**Important**: All dataloaders keep data on CPU and let PyTorch Lightning handle GPU transfer automatically.

---

## 🚀 Training Scripts

### Shell Scripts Overview

The `shell_scripts/` directory contains bash scripts for training experiments.

### **Pretraining Scripts**

#### `run_l1_pretrain.sh` - L1 Supervised Pretraining
Trains a model on the L1 dataset (7 classes) for supervised pretraining using `train_l1_baseline.py`.


**Output**: Checkpoint saved in `./CerraData-4MM/experiment_results/weights/l1_pretrain_*/last.ckpt`

**Usage:**
```bash
./shell_scripts/run_l1_pretrain.sh
```

---

#### `run_moco_pretrain.sh` - MoCo Self-Supervised Pretraining
Trains a MoCo v2 encoder using self-supervised contrastive learning on unlabeled multimodal data via `train_moco_lightly.py`.


**⚠️ Note**: In my experience, MoCo training only worked reliably with `--norm z_score`, `--temperature 0.07`, and `--memory_bank_size 65536`. Other normalization types caused training instability or dimensional collapse.

**Output**: Encoder saved in `.../experiment_results/weights/moco_*/encoder_final.pth`



---

### **L2 Training Loop Scripts**

These scripts train models on varying percentages of L2 labeled data to evaluate data efficiency.

**Data Percentages Tested**: `0.5%, 1%, 2.5%, 3.5%, 5%, 10%, 25%, 50%`

---

#### `run_l2_baseline_loop.sh` - L2 Baseline (No Pretraining, Random Start)
Trains L2 models from **random initialization** (no pretraining) using `train_l2_baseline.py`.

**Loop Configuration** (from `shell_scripts/run_l2_baseline_loop.sh`):
```bash
PERCENTAGES="0.5 1 2.5 3.5 5 10 25 50"   # Data percentages to test
PATIENCE=50                               # Early stopping patience
LOG_BASE=".../experiment_diff_percentages/logs/baseline"
CHECKPOINT_BASE=".../experiment_diff_percentages/weights/baseline"
```



**Output Structure:**
```
experiment_diff_percentages/
├── logs/baseline/
│   ├── l2_baseline_14classes_multimodal_0_5percent_*/
│   ├── l2_baseline_14classes_multimodal_1percent_*/
│   └── ... (one per percentage)
└── weights/baseline/
    └── ... (same structure)
```

**Usage:**
```bash
./shell_scripts/run_l2_baseline_loop.sh
```

---

#### `run_l2_finetune_loop.sh` - L2 Fine-tuning (L1-Pretrained, Frozen Encoder)
Trains L2 models by fine-tuning from **L1 pretrained checkpoint** with **frozen encoder** using `train_l2_finetune.py`.

**Before Running**:
1. Complete L1 pretraining using `run_l1_pretrain.sh`
2. Update `L1_CHECKPOINT` path in script (line 7):
   ```bash
   L1_CHECKPOINT="/path/to/l1_pretrain_*/last.ckpt"
   ```

**Loop Configuration** (from `shell_scripts/run_l2_finetune_loop.sh`):
```bash
PERCENTAGES="0.5 1 2.5 3.5 5 10 25 50"
PATIENCE=50
LOG_BASE=".../experiment_diff_percentages/logs/finetune_frozen"
CHECKPOINT_BASE=".../experiment_diff_percentages/weights/finetune_frozen"
```



**Output Structure:**
```
experiment_diff_percentages/
├── logs/finetune_frozen/
│   ├── l2_finetune_14classes_frozenenc_0_5percent_*/
│   └── ...
└── weights/finetune_frozen/
    └── ...
```

**Usage:**
```bash
# 1. Update L1_CHECKPOINT path in script
# 2. Run:
./shell_scripts/run_l2_finetune_loop.sh
```

---

#### `run_l2_from_supervision_loop.sh` - L2 from MoCo (Self-Supervised Pretrained, Frozen Encoder)
Trains L2 models using **MoCo pretrained encoder** with **frozen encoder** via `train_l2_from_simclr.py`.

**Before Running**:
1. Complete MoCo pretraining using `run_moco_pretrain.sh`
2. Update `MOCO_ENCODER` path in script (line 8):
   ```bash
   MOCO_ENCODER="/path/to/moco_pretrain_*/encoder_final.pth"
   ```

**Loop Configuration** (from `shell_scripts/run_l2_from_supervision_loop.sh`):
```bash
PERCENTAGES="0.5 1 2.5 3.5 5 10 25 50"
PATIENCE=50
LOG_BASE=".../experiment_diff_percentages/logs/frozen_moco_encoder"
CHECKPOINT_BASE=".../experiment_diff_percentages/weights/frozen_moco_encoder"
```



**⚠️ Note**: Must use `--norm z_score` to match the MoCo encoder's pretraining normalization. Mismatched normalization may cause problems.

**Output Structure:**
```
experiment_diff_percentages/
├── logs/frozen_moco_encoder/
│   ├── l2_from_moco_14ch_frozen_0_5percent_*/
│   └── ...
└── weights/frozen_moco_encoder/
    └── ...
```

**Usage:**
```bash
# 1. Update MOCO_ENCODER path in script
# 2. Run:
./shell_scripts/run_l2_from_supervision_loop.sh
```

---

## 📊 Results Analysis

### Extract Results from TensorBoard Logs

Use the provided script `extract_and_plot_results.py` to extract test F1-macro scores and create comparison plots:

```bash
python extract_and_plot_results.py
```

**Outputs:**
1. `l2_results_comparison.md` - Markdown tables with F1 scores for each experiment
2. `l2_comparison_plot.png` - Comparison plot (PNG, 300 DPI)
3. `l2_comparison_plot.pdf` - Publication-quality plot (PDF)

The script automatically reads TensorBoard logs from:
```
experiment_diff_percentages/logs/
├── baseline/
├── finetune_frozen/
└── frozen_moco_encoder/
```

---

## 🔬 Complete Experiment Workflow

### Step-by-Step Training Pipeline

0. **Dataset Preparation** (First time only):
   ```bash
   python CerraData-4MM/split_dataset.py
   ```
   Splits dataset into train/val/test (70%/15%/15%)

1. **L1 Supervised Pretraining**:
   ```bash
   ./shell_scripts/run_l1_pretrain.sh
   ```
   Wait for completion (~500 epochs with early stopping)

2. **MoCo Self-Supervised Pretraining** (parallel with step 1):
   ```bash
   ./shell_scripts/run_moco_pretrain.sh
   ```
   Wait for completion (~1000 epochs)

3. **L2 Baseline** (No Pretraining):
   ```bash
   ./shell_scripts/run_l2_baseline_loop.sh
   ```
   Trains 8 models (one per data percentage)

4. **L2 Fine-tuning** (From L1 Pretraining):
   ```bash
   # Update L1_CHECKPOINT in script first!
   ./shell_scripts/run_l2_finetune_loop.sh
   ```
   Trains 8 models with frozen L1 encoder

5. **L2 from MoCo** (From Self-Supervised Pretraining):
   ```bash
   # Update MOCO_ENCODER in script first!
   ./shell_scripts/run_l2_from_supervision_loop.sh
   ```
   Trains 8 models with frozen MoCo encoder

6. **Extract and Compare Results**:
   ```bash
   python extract_and_plot_results.py
   ```

---

## ⚙️ Key Configuration Notes

### Normalization

- **Default**: `z_score` normalization is used across all training scripts
- **MoCo Requirement**: MoCo pretraining and fine-tuning work reliably with `--norm z_score` (other options caused issues in testing)
- **Available Options**: `"none"`, `"0to1"`, `"1to1"`, `"z_score"`

### Data Percentages

All loop scripts test: **0.5%, 1%, 2.5%, 3.5%, 5%, 10%, 25%, 50%**

### Early Stopping

- L1 Pretraining (`run_l1_pretrain.sh`): `--patience 30`
- L2 Training Loops: `--patience 50`

### Frozen vs. Trainable Encoder

- **Frozen** (`--freeze_encoder`): Only decoder/classifier trains - used in loop scripts
- **Trainable** (no flag): Full model trains - use for end-to-end fine-tuning

---

## 📁 Project Structure

```
CerraData_Project_Phenorob/
├── dataset_loader_official/
│   ├── dataset_loader.py        # L2 dataloaders (14 classes)
│   └── dataset_loader_7.py      # L1 dataloaders (7 classes)
├── shell_scripts/
│   ├── run_l1_pretrain.sh              # L1 supervised pretraining
│   ├── run_moco_pretrain.sh            # MoCo self-supervised pretraining
│   ├── run_l2_baseline_loop.sh         # L2 baseline (random init)
│   ├── run_l2_finetune_loop.sh         # L2 from L1 (frozen encoder)
│   └── run_l2_from_supervision_loop.sh # L2 from MoCo (frozen encoder)
├── train_l1_baseline.py         # L1 training script
├── train_l2_baseline.py         # L2 baseline training script
├── train_l2_finetune.py         # L2 fine-tuning script
├── train_l2_from_simclr.py      # L2 from MoCo/SimCLR script
├── train_moco_lightly.py        # MoCo pretraining script
├── extract_and_plot_results.py  # Results extraction and plotting
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🎯 Research Questions

This codebase investigates:

1. **Transfer Learning**: Does L1 pretraining improve L2 classification?
2. **Self-Supervised Learning**: Can MoCo pretraining match or exceed supervised pretraining?
3. **Data Efficiency**: How much labeled L2 data is needed with different pretraining strategies?
4. **Multimodal Fusion**: What is the benefit of combining SAR + MSI vs. single modality?

---

## 🔧 Troubleshooting

### Common Issues

1. **GDAL Installation Error**:
   ```bash
   sudo apt-get install gdal-bin libgdal-dev python3-gdal
   pip install gdal==$(gdal-config --version)
   ```

2. **MoCo Training Instability**:
   - Ensure `--norm z_score` is used in `run_moco_pretrain.sh`
   - Check `--temperature 0.07` (low temperature helps prevent collapse)
   - Verify `--memory_bank_size 65536` (larger memory bank recommended)

3. **MoCo Fine-tuning Poor Performance**:
   - Ensure `--norm z_score` matches MoCo pretraining in `run_l2_from_supervision_loop.sh`
   - Verify correct encoder path in `MOCO_ENCODER` variable

4. **Out of Memory**:
   - Reduce `--batch_size` in scripts
   - Reduce `--num_workers` in MoCo script
   - Use gradient accumulation (modify training scripts)

5. **TensorBoard Logs Missing**:
   - Check `LOG_BASE` directories exist (scripts create them automatically)
   - Verify experiment names match in scripts
   - Ensure training completed at least one epoch

---

**Happy Training! 🚀**
