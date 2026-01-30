#!/bin/bash
#SBATCH --job-name=l1_pretrain
#SBATCH --output=logs/l1_pretrain_%j.out
#SBATCH --error=logs/l1_pretrain_%j.err

echo "=== L1 PRETRAINING EXPERIMENT ==="
echo "Starting at: $(date)"

python train_l1_baseline.py \
    --data_dir /home/s52melba/CerraData_Project_Phenorob/CerraData-4MM/dataset_splitted \
    --experiment_name "l1_pretrain_7classes_multimodal" \
    --gpu_ids "0" \
    --batch_size 256 \
    --num_epochs 500 \
    --learning_rate 1e-4 \
    --checkpoint_dir ./CerraData-4MM/experiment_results/weights \
    --log_dir ./CerraData-4MM/experiment_results/logs \
    --patience 30 \
    --seed 42

echo "Completed at: $(date)"