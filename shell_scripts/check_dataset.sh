#!/bin/bash

echo "=== CHECKING DATASET STRUCTURE ==="
echo "Date: $(date)"
echo ""

DATASET_PATH="/home/s52melba/CerraData_Project_Phenorob/kaggle_temp/cerradata_4mm"

echo "1. Checking main dataset directory:"
echo "Path: $DATASET_PATH"
ls -la "$DATASET_PATH/"
echo ""

echo "2. Checking what label directories exist:"
ls -la "$DATASET_PATH"/semantic_*/ 2>/dev/null || echo "No semantic_* directories found"
echo ""

echo "3. Checking for specific image ID that failed (42312):"
find "$DATASET_PATH/" -name "*42312*" 2>/dev/null || echo "No files with ID 42312 found"
echo ""

echo "4. File counts in each directory:"
echo "MSI images:"
ls "$DATASET_PATH/msi_images/" 2>/dev/null | wc -l || echo "msi_images directory not found"

echo "14-class labels (semantic_14c):"
ls "$DATASET_PATH/semantic_14c/" 2>/dev/null | wc -l || echo "semantic_14c directory not found"

echo "7-class labels (semantic_7c):"
ls "$DATASET_PATH/semantic_7c/" 2>/dev/null | wc -l || echo "semantic_7c directory not found"

echo ""
echo "5. Sample files from each directory:"
echo "Sample MSI images (first 3):"
ls "$DATASET_PATH/msi_images/" 2>/dev/null | head -3 || echo "No MSI images found"

echo "Sample 14-class labels (first 3):"
ls "$DATASET_PATH/semantic_14c/" 2>/dev/null | head -3 || echo "No 14-class labels found"

echo "Sample 7-class labels (first 3):"
ls "$DATASET_PATH/semantic_7c/" 2>/dev/null | head -3 || echo "No 7-class labels found"

echo ""
echo "=== CHECK COMPLETE ==="