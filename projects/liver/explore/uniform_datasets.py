import os
import nibabel as nib

folder_datasets = 'E:/Datasets_NEW'
subfolders = ['liver', 'vessels', 'tumors', 'segments']

for subfolder in subfolders:
    subfolder_path = os.path.join(folder_datasets, subfolder)
    print('Subfolder Path = {}'.format(subfolder_path))
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

#############
### LIVER ###
#############

##############
### SLIVER ###
##############