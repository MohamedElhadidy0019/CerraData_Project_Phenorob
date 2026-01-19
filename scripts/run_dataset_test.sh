#!/bin/bash

echo "=== Testing Dataset Functionality ==="
echo "Starting at: $(date)"

python test_dataset_functionality.py \
    --data_dir /home/s52melba/CerraData_Project_Phenorob/kaggle_temp/cerradata_4mm \
    --label_level L1 \
    --batch_size 4 \
    --device cuda:0

if [ $? -eq 0 ]; then
    echo "✓ Script completed successfully at: $(date)"
else
    echo "✗ Script failed with error at: $(date)"
fi