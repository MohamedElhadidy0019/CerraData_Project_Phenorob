# Checkpoints for Data Scaling Experiments

This directory contains model checkpoints for experiments testing different data percentages (0.1% - 50%).

## Structure

- **baseline/** - Random initialization → L2 training
- **finetuning/** - L1 pretrain → L2 fine-tuning
- **self_supervised/** - SimCLR/MoCo pretrain → L2 fine-tuning

Each subdirectory contains checkpoints for all tested data percentages.
