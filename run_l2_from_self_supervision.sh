#!/bin/bash

# UPDATE THIS PATH after SimCLR pretraining completes!
# The encoder will be saved in: checkpoints_data_splitted/simclr_pretrain_resnet34_TIMESTAMP/encoder_final.pth
MOCO_ENCODER="./checkpoints_data_splitted/moco_pretrain_resnet34_v2_more_aggressive_20251218_114641/encoder_final.pth"
# checkpoints_data_splitted/moco_pretrain_resnet34_v2_more_aggressive_20251218_114641/encoder_final.pth

echo "=== L2 FINE-TUNING FROM MOCO PRETRAINING ==="
echo "Using MOCO encoder: $MOCO_ENCODER"
echo "Starting at: $(date)"

python train_l2_from_simclr.py \
    --moco_encoder "$MOCO_ENCODER" \
    --data_dir /home/s52melba/CerraData_Project_Phenorob/cerradata_splitted \
    --experiment_name "l2_from_moco_aggressive_5percent" \
    --gpu_ids "0" \
    --batch_size 100 \
    --num_epochs 500 \
    --learning_rate 1e-4 \
    --data_percentage 5 \
    --patience 40 \
    --num_workers 4 \
    --checkpoint_dir ./checkpoints_data_splitted \
    --log_dir ./logs_splitted

echo "Completed at: $(date)"
