# CerraData Project - Step 2: Baseline Training

This implements **Step 2** of your project: Training with Random Initialization (no pretraining) for 14-class semantic segmentation.

## What's Implemented

✅ **U-Net Model**: Using segmentation-models-pytorch with ResNet34 encoder  
✅ **Random Initialization**: No pretrained weights (`encoder_weights=None`)  
✅ **14-class Segmentation**: L2 labels for fine-grained land cover classification  
✅ **Cross-Entropy Loss**: With softmax activation (built into PyTorch's CrossEntropyLoss)  
✅ **70/15/15 Data Split**: Automatic train/validation/test splitting  
✅ **Multi-GPU Support**: Specify GPU IDs for single or multi-GPU training  

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
python download_data.py
```

### 3. Run Baseline Training

**Single GPU (GPU 0):**
```bash
python train_baseline.py --gpu_ids "0" --batch_size 16 --num_epochs 50
```

**Multi-GPU (GPUs 0,1,2,3):**
```bash
python train_baseline.py --gpu_ids "0,1,2,3" --batch_size 64 --num_epochs 50
```

**CPU only:**
```bash
python train_baseline.py --batch_size 8 --num_epochs 50
```

### 4. Full Command Line Options
```bash
python train_baseline.py \
    --data_dir ./data \
    --batch_size 16 \
    --num_epochs 100 \
    --learning_rate 1e-3 \
    --num_workers 4 \
    --gpu_ids "0,1" \
    --checkpoint_dir ./checkpoints \
    --log_dir ./logs
```

## Model Architecture

- **Input**: 12-channel Sentinel-2 multispectral images (128×128)
- **Model**: U-Net with ResNet34 encoder (random initialization)
- **Output**: 14-class segmentation masks
- **Loss**: CrossEntropyLoss (includes softmax)
- **Optimizer**: Adam with ReduceLROnPlateau scheduler

## 14 Classes (L2 Labels)

| ID | Class | Abbreviation |
|----|-------|--------------|
| 0  | Pasture | Pa |
| 1  | Primary Natural Vegetation | V1 |
| 2  | Secondary Natural Vegetation | V2 |
| 3  | Water body | Wt |
| 4  | Mining | Mg |
| 5  | Urban area | UA |
| 6  | Other Built area | OB |
| 7  | Forestry | Ft |
| 8  | Perennial Agriculture | PR |
| 9  | Semi-perennial Agriculture | SP |
| 10 | Temporary agriculture of 1 cycle | T1 |
| 11 | Temporary agriculture of 1+ cycle | T1+ |
| 12 | Other Uses | OU |
| 13 | Deforestation 2022 | Df |

## Output Files

- **Checkpoints**: `./checkpoints/baseline_l2_14class_[timestamp]/`
- **Logs**: `./logs/baseline_l2_14class_[timestamp]/`
- **Training Summary**: `training_summary.txt` in checkpoint directory

## Metrics Tracked

- Accuracy, F1-score (macro/weighted), IoU (macro)
- Per-class F1 and IoU scores
- Training/validation loss curves

This baseline will serve as comparison for your subsequent pretraining experiments.