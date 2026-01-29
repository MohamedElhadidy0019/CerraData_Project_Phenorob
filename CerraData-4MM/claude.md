  What the dataset is                                                                     
                                                                                          
  CerraData-4MM is a multimodal land use/land cover dataset from the Cerrado biome        
  (Brazil). It has:                                                                       
  - 30,291 patches (128x128, 10m resolution)                                              
  - 2 modalities: MSI (Sentinel-2, 12 channels) + SAR (Sentinel-1, 2 channels)            
  - 2 class hierarchies: L1 (7 classes) and L2 (14 classes)                               
  - 6 subfolders in raw data: msi_images/, sar_images/, semantic_7c/, semantic_14c/,      
  edge_7c/, edge_14c/                                                                     
                                                                                          
  Where the data lives                                                                    
                                                                                          
  - Raw download: kaggle_temp/cerradata_4mm/ — all 30,291 files per folder, complete      
  - Authors' repo clone: CerraData-4MM/ — contains loaders, experiments code, and a       
  dataset_splitted/ from splitfolders                                                     
  - Your custom splits: cerradata_splitted/ — created by your create_physical_splits.py,  
  only MSI + L1/L2 labels (no SAR, no edges)                                              
                                                                                          
  Authors' loader files (in CerraData-4MM/CerraData-4MM Experiments/util/)                
                                                                                          
  - dataset_loader.py → loads L2 (14 classes) — reads semantic_14c/, edge_14c/            
  - dataset_loader_7.py → loads L1 (7 classes) — reads semantic_7c/, edge_7c/             
  - dataset_loader_4test.py → L1, same but also returns file paths for inference          
  - Dataset classes in each: MMDataset (MSI+SAR, 14ch), MSIDataset (12ch), SARDataset     
  (2ch), MM2Dataset (14ch + distance transform edges)                                     
  - Normalization options: 'none', '0to1' (min-max), '1to1' (log-based)                   
                                                                                          
  Bugs found in YOUR files                                                                
                                                                                          
  create_physical_splits.py:                                                              
  - Only copies msi_images → no SAR, no edges                                             
  - Uses sklearn.train_test_split instead of splitfolders.ratio (different splits than    
  original)                                                                               
                                                                                          
  dataset.py:                                                                             
  1. CRITICAL — L1 labels corrupted: Loads from labels_l1/ (already 7-class values 0-6)   
  then applies l2_to_l1 mapping on them. This scrambles 5/7 classes (Agriculture→Forest,  
  Mining→Water, Building→Mining, Water→Building, OtherUses→Building)                      
  2. Wrong mapping: Forestry (Ft, L2 ID 7) mapped to Forest (L1=1) instead of Agriculture 
  (L1=2)                                                                                  
                                                                                          
  State of CerraData-4MM/dataset_splitted/ (authors' split)                               
  ┌──────────────┬─────────────────────┬─────────┬─────────┐                              
  │    Folder    │        train        │   val   │  test   │                              
  ├──────────────┼─────────────────────┼─────────┼─────────┤                              
  │ msi_images   │ 21,203              │ 4,543   │ 4,545   │                              
  ├──────────────┼─────────────────────┼─────────┼─────────┤                              
  │ sar_images   │ 21,203              │ present │ present │                              
  ├──────────────┼─────────────────────┼─────────┼─────────┤                              
  │ semantic_7c  │ 21,203              │ 4,543   │ present │                              
  ├──────────────┼─────────────────────┼─────────┼─────────┤                              
  │ semantic_14c │ 11,896 (INCOMPLETE) │ MISSING │ MISSING │                              
  ├──────────────┼─────────────────────┼─────────┼─────────┤                              
  │ edge_7c      │ 21,203              │ present │ present │                              
  ├──────────────┼─────────────────────┼─────────┼─────────┤                              
  │ edge_14c     │ 21,203              │ present │ present │                              
  └──────────────┴─────────────────────┴─────────┴─────────┘                              
  - L1 is usable (all folders complete)                                                   
  - L2 is broken (semantic_14c split was interrupted — only 11,896/21,203 in train,       
  missing from val+test)                                                                  
                                                                                          
  Naming mismatch                                                                         
                                                                                          
  Authors' loaders hardcode opt_images/ but the Kaggle download and splits use            
  msi_images/. Fix with symlinks:                                                         
  ln -s msi_images opt_images  # in each split dir                                        
                                                                                          
  Script I wrote                                                                          
                                                                                          
  CerraData-4MM/check_splits.py — counts all .tif files per subfolder per split, compares 
  to raw source, checks consistency and overlap. Run with python                          
  CerraData-4MM/check_splits.py.   