#!/usr/bin/env python3
"""
Download CerraData-4MM dataset from Kaggle
"""
import os
import kagglehub
from pathlib import Path

def download_cerradata():
    """Download the CerraData-4MM dataset"""
    print("Downloading CerraData-4MM dataset...")
    
    # Download to current directory
    data_path = kagglehub.dataset_download("cerranet/cerradata-4mm")
    print(f"Dataset downloaded to: {data_path}")
    
    # Create symlink in project directory for easier access
    project_data_dir = Path("./data")
    if not project_data_dir.exists():
        os.symlink(data_path, project_data_dir)
        print(f"Created symlink: {project_data_dir} -> {data_path}")
    
    return data_path

if __name__ == "__main__":
    data_path = download_cerradata()
    print(f"Data ready at: {data_path}")