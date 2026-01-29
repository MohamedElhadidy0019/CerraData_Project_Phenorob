import splitfolders

# Split data into Train and testing
# /home/s52melba/CerraData_Project_Phenorob/kaggle_temp/cerradata_4mm
input_folder = '/home/s52melba/CerraData_Project_Phenorob/kaggle_temp/cerradata_4mm/'
output_folder = '/home/s52melba/CerraData_Project_Phenorob/CerraData-4MM/dataset_splitted/'

# Ratio of split are in order of train/val/test.
splitfolders.ratio(input_folder, output_folder, seed=42, ratio=(0.7, .15, .15))