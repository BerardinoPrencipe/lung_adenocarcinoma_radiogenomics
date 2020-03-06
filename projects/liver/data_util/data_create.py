#python3 projects/liver/data_util/data_create.py

import numpy as np
import nibabel as nib
import os
import platform
import sys

current_path_abs = os.path.abspath('.')
sys.path.append(current_path_abs)
print('{} appended to sys!'.format(current_path_abs))

from util.utils_calc import normalize_data, get_patient_id
from projects.liver.train.config import window_hu

### variables ###

# validation list
val_list = [idx for idx in range(10)]

# source folder where the .nii.gz files are located
dataset_folder = os.path.join(current_path_abs, 'datasets/LiTS')
source_folder = os.path.join(dataset_folder, 'nii/train')

#################
LIVER_CLASS = 1
TUMOR_CLASS = 2

# destination folder where the subfolders with npy files will go
destination_folder = os.path.join(dataset_folder, 'npy')
destination_folder_no_norm = os.path.join(dataset_folder, 'npy_no_norm')

destination_folders = list()
destination_folders.append(destination_folder)
destination_folders.append(destination_folder_no_norm)

# create destination folder and possible subfolders
subfolders = ["train", "val"]
for dest_folder in destination_folders:
    if not os.path.isdir(dest_folder):
        os.makedirs(dest_folder)
    for name in subfolders:
        if not os.path.isdir(os.path.join(dest_folder, name)):
            os.makedirs(os.path.join(dest_folder, name))

for idx, file_name in enumerate(os.listdir(source_folder)):

    print('Iter {} on {}'.format(idx, len(os.listdir(source_folder))-1))
    print(file_name)

    # create new file name by stripping .nii and adding .npy
    new_file_name = file_name[:-4]

    # decide whether it will go to the train or val folder
    sub = subfolders[1] if get_patient_id(file_name) in val_list else subfolders[0]

    # load file
    data = nib.load(os.path.join(source_folder, file_name))

    # convert to numpy
    data = data.get_data()

    # check if it is a volume file and clip and standardize if so
    if file_name[:3] == 'vol':
        print('[Before Normalization] data.min() = {} data.max() = {}'.format(data.min(), data.max()))
        norm_data = normalize_data(data, window_hu)
        print('[After  Normalization] data.min() = {} data.max() = {}'.format(norm_data.min(),norm_data.max()))
        norm_data = np.transpose(norm_data, (2,0,1))

    # check if it is a segmentation file and select only the tumor (2) as positive label
    # if file_name[:3] == 'seg': data = (data==LIVER_CLASS).astype(np.uint8)
    if file_name[:3] == 'seg': data = (data>0).astype(np.uint8)

    # transpose so the z-axis (slices) are the first dimension
    data = np.transpose(data, (2, 0, 1))

    # Save images and masks with no norm
    for i, z_slice in enumerate(data):
        # save at new location (train or val)
        np.save(os.path.join(destination_folder_no_norm, sub, new_file_name + '_' + str(i)), z_slice)

    # Save masks with norm
    if file_name[:3] == 'seg':
        for i, z_slice in enumerate(data):
            np.save(os.path.join(destination_folder, sub, new_file_name + '_' + str(i)), z_slice)

    # Save volumes with norm
    if file_name[:3] == 'vol':
        for i, z_slice in enumerate(norm_data):
            np.save(os.path.join(destination_folder, sub, new_file_name + '_' + str(i)), z_slice)