# CerraData Project - Phenorob

Semantic segmentation for multispectral Sentinel-2 satellite imagery using the CerraData-4MM dataset.

## Quick Start

```bash
# Download and prepare data
python download_data.py
python create_physical_splits.py

# Run training (example: L2 baseline)
bash scripts/run_l2_baseline.sh
```

## Project Structure

```
├── dataset.py                 # Dataset loading and transforms
├── model.py                   # U-Net architecture with ResNet34 encoder
├── train_baseline.py          # Supervised training from scratch
├── train_l2_finetune.py       # Fine-tune L2 from L1 pretrained model
├── train_simclr_lightly.py    # SimCLR self-supervised pretraining
├── train_moco_lightly.py      # MoCo v2 self-supervised pretraining
├── train_l2_from_simclr.py    # Fine-tune L2 from self-supervised encoder
├── inference.py               # Run inference on test set
├── test_model.py              # Evaluate trained models
├── scripts/                   # Shell scripts for running experiments
│   ├── run_l1_pretrain.sh
│   ├── run_l2_baseline.sh
│   ├── run_simclr_pretrain.sh
│   └── ...
├── checkpoints_*/             # Saved model checkpoints
└── logs_*/                    # Training logs
```

## Three Training Approaches

| Approach | Description | Script |
|----------|-------------|--------|
| **Baseline** | Train from random init directly on L2 | `train_baseline.py` |
| **Hierarchical** | Pretrain on L1 (7 classes) → Fine-tune on L2 (14 classes) | `train_l2_finetune.py` |
| **Self-Supervised** | SimCLR/MoCo pretraining → Fine-tune on L2 | `train_l2_from_simclr.py` |

## Label Hierarchy

- **L1 (Coarse)**: 7 classes - Pasture, Forest, Agriculture, Mining, Building, Water body, Other
- **L2 (Fine-grained)**: 14 classes

## Architecture

- **Model**: U-Net with ResNet34 encoder (segmentation-models-pytorch)
- **Input**: 12-channel Sentinel-2 images (128x128 pixels)
- **Dataset**: CerraData-4MM


## Data Scaling Experiments

The project supports experiments across data percentages: 0.1%, 0.5%, 1%, 2.5%, 5%, 10%, 25%, 50%, 100%

```bash
# Run all percentages for baseline
bash scripts/run_l2_baseline_loop.sh

# Run all percentages for hierarchical fine-tuning
bash scripts/run_l2_finetune_loop.sh

# Run all percentages for self-supervised fine-tuning
bash scripts/run_l2_from_supervision_loop.sh
```
