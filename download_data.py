#!/usr/bin/env python3
"""
Download CerraData-4MM dataset from Kaggle
"""
import os
from pathlib import Path

# Set Kaggle config directory to custom location
os.environ["KAGGLE_CONFIG_DIR"] = "/scratch/s52melba/CerraData_Project_Phenorob"

from kaggle.api.kaggle_api_extended import KaggleApi


def download_cerradata():
    """Download the CerraData-4MM dataset"""
    download_dir = Path("/scratch/s52melba/CerraData_Project_Phenorob/kaggle_temp")
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading CerraData-4MM dataset to: {download_dir}")
    print(f"Download directory exists: {download_dir.exists()}")
    print(f"Download directory is writable: {os.access(download_dir, os.W_OK)}")
    
    # Initialize Kaggle API
    print("Initializing Kaggle API...")
    api = KaggleApi()
    
    print("Authenticating...")
    api.authenticate()
    print("Authentication successful!")
    
    # Download dataset with explicit path
    print("Starting download (this may take a while for large datasets)...")
    print("Note: The download may appear to hang, but it's working. Monitor with:")
    print(f"  watch -n 5 'du -sh {download_dir}'")
    
    try:
        api.dataset_download_files(
            "cerranet/cerradata-4mm",
            path=str(download_dir),
            unzip=True,
            quiet=False  # Show progress if available
        )
        print(f"\nDownload complete!")
    except Exception as e:
        print(f"ERROR during download: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # List what's in the directory
    print(f"\nContents of {download_dir}:")
    items = list(download_dir.iterdir())
    if items:
        for item in items:
            size_mb = item.stat().st_size / (1024**2)
            print(f"  - {item.name} ({size_mb:.2f} MB)")
    else:
        print("  (empty)")
    
    return download_dir


if __name__ == "__main__":
    data_path = download_cerradata()
    if data_path:
        print(f"\nData ready at: {data_path}")
    else:
        print("\nDownload failed!")