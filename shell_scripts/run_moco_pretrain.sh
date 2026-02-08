#!/bin/bash

echo "=== MOCO V2 SELF-SUPERVISED PRETRAINING (14-channel Multimodal) ==="
echo "Starting at: $(date)"

# Stronger augmentations + larger memory bank + lower temperature
# Following recommendations to address dimensional collapse
# Now using 14 channels (12 MSI + 2 SAR) from CerraData-4MM
python train_moco_lightly.py \
    --data_dir /home/s52melba/CerraData_Project_Phenorob/CerraData-4MM/dataset_splitted \
    --experiment_name "moco_pretrain_14ch_multimodal_aggressive" \
    --gpu_ids "0" \
    --batch_size 256 \
    --num_epochs 1000 \
    --learning_rate 0.05 \
    --temperature 0.07 \
    --projection_dim 128 \
    --memory_bank_size 65536 \
    --num_workers 32 \
    --prefetch_factor 2 \
    --checkpoint_dir /home/s52melba/CerraData_Project_Phenorob/CerraData-4MM/experiment_results/weights/ \
    --log_dir /home/s52melba/CerraData_Project_Phenorob/CerraData-4MM/experiment_results/logs/

echo "Completed at: $(date)"
