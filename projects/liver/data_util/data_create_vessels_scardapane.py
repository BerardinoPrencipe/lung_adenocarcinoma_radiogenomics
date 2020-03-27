import os
import numpy as np
import nibabel as nib
import sys

current_path_abs = os.path.abspath('.')
sys.path.append(current_path_abs)
print('{} appended to sys!'.format(current_path_abs))

from utils.utils_calc import normalize_data, get_patient_id
from projects.liver.train.config import window_hu

#%%
val_list = ['ct_mask_00.nii.gz', 'ct_scan_00.nii.gz']

do_one_class_only = True

dataset_dir = 'H:/Datasets/Liver/LiverScardapaneNew'
source_dir = os.path.join(dataset_dir, 'nii')
if do_one_class_only:
    destination_dir = os.path.join(dataset_dir, 'npy_one_class')
else:
    destination_dir = os.path.join(dataset_dir, 'npy')

# create destination folder and possible subfolders
subfolders = ["train", "val"]

for name in subfolders:
    if not os.path.isdir(os.path.join(destination_dir, name)):
        os.makedirs(os.path.join(destination_dir, name))
        print('Created subdir: {}'.format(name))

print('Source Folder  = {}'.format(source_dir))

#%%
for idx, file_name in enumerate(os.listdir(source_dir)):

    print('Iter {} on {}'.format(idx, len(os.listdir(source_dir))-1))
    print(file_name)

    # create new file name by stripping .nii.gz and adding .npy
    new_file_name = file_name[3:-7]
    new_file_name = new_file_name.replace('_', '-')

    # decide whether it will go to the train or val folder
    sub = subfolders[1] if file_name in val_list else subfolders[0]

    # load file
    data = nib.load(os.path.join(source_dir, file_name))

    # convert to numpy
    data = data.get_data()

    # check if it is a volume file and clip and standardize if so
    if file_name[:4] == 'ct_s':
        print('[Before Normalization] data.min() = {} data.max() = {}'.format(data.min(), data.max()))
        data = normalize_data(data, window_hu)
        print('[After  Normalization] data.min() = {} data.max() = {}'.format(data.min(),data.max()))


    # check if it is a segmentation file and select only the tumor (2) as positive label
    if do_one_class_only:
        if file_name[:4] == 'ct_m': data = (data>0).astype(np.uint8)
    else:
        if file_name[:4] == 'ct_m': data = data.astype(np.uint8)

    # transpose so the z-axis (slices) are the first dimension
    data = np.transpose(data, (2, 0, 1))

    # Save images and masks with no norm
    for i, z_slice in enumerate(data):
        # save at new location (train or val)
        np.save(os.path.join(destination_dir, sub, new_file_name + '_' + str(i)), z_slice)

