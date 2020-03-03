import numpy as np
import nibabel as nib
import os
import sys
import platform

current_path_abs = os.path.abspath('.')
sys.path.append(current_path_abs)
print('{} appended to sys!'.format(current_path_abs))

from utils_calc import normalize_data, get_patient_id
from projects.liver.train.config import window_hu

# Use local path or absolute
if 'Ubuntu' in platform.system() or 'Linux' in platform.system():
    isLinux = True
else:
    isLinux = False

### variables ###


# validation list
val_list = ["{:02d}".format(idx) for idx in range(1,3)]
exclude_list = [] # exclude_list = ["10"]

dataset_folder = os.path.join(current_path_abs, 'datasets/ircadb')
# source folder where the .nii.gz files are located
source_folder = os.path.join(dataset_folder, 'nii')

#################

LIVER_CLASS = (1,)
VESSELS_CLASS = (2, 3)
ARTERY_CLASS = (4,)

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
        print('Created destionation_folder in {}'.format(dest_folder))

    for name in subfolders:
        if not os.path.isdir(os.path.join(dest_folder, name)):
            os.makedirs(os.path.join(dest_folder, name))
            print('Created subdir: {}'.format(name))


source_subolders = os.listdir(source_folder)
source_subolders.sort()
print('Source Folder  = {}'.format(source_folder))
print('Source SubDirs = {}'.format(source_subolders))

do_no_norm = False
do_norm    = False

for idx, subfolder_source in enumerate(source_subolders):

    print('Index {} on {}'.format(idx, len(os.listdir(source_folder))-1))

    image_folder = os.path.join(source_folder, subfolder_source, 'image')
    mask_folder = os.path.join(source_folder, subfolder_source, 'mask')

    image_filename = os.path.join(image_folder, 'image.nii')
    mask_filename = os.path.join(mask_folder, 'mask.nii')

    print("Image: ", image_filename)
    print("Mask : ", mask_filename)

    id_patient = subfolder_source[-2:]
    # create new file name by stripping .nii and adding .npy
    new_image_filename = "volume-{}".format(id_patient)
    new_mask_filename = "segmentation-{}".format(id_patient)

    if id_patient in exclude_list:
        print('Excluding patient {}'.format(id_patient))
    else:
        # decide whether it will go to the train or val folder
        sub = subfolders[1] if id_patient in val_list else subfolders[0]

        # load file
        image_data = nib.load(image_filename)
        mask_data = nib.load(mask_filename)

        # convert to numpy
        mask_data = mask_data.get_data()
        mask_data_hv = (mask_data == VESSELS_CLASS[0]).astype(np.uint8)
        mask_data_pv = (mask_data == VESSELS_CLASS[1]).astype(np.uint8)
        mask_data          = np.logical_or(mask_data_hv, mask_data_pv)
        image_data_no_norm = image_data.get_data()
        image_data_norm    = normalize_data(image_data_no_norm, window_hu)

        # transpose so the z-axis (slices) are the first dimension
        image_data_norm    = np.transpose(image_data_norm, (2,0,1))
        image_data_no_norm = np.transpose(image_data_no_norm, (2, 0, 1))
        mask_data          = np.transpose(mask_data, (2,0,1))

        print(f'[  Norm   ] Max = {image_data_norm.max()} - Min = {image_data_norm.min()}')
        print(f'[ No Norm ] Max = {image_data_no_norm.max()} - Min = {image_data_no_norm.min()}')

        # loop through the mask slices
        for i, z_slice in enumerate(mask_data):
            # save at new location (train or val)
            if do_norm:
                np.save(os.path.join(destination_folder, sub, new_mask_filename + '_' + str(i)), z_slice)
            if do_no_norm:
                np.save(os.path.join(destination_folder_no_norm, sub, new_mask_filename + '_' + str(i)), z_slice)

        if do_norm:
            # loop through the scan slices
            for i, z_slice in enumerate(image_data_norm):
                # save at new location (train or val)
                np.save(os.path.join(destination_folder, sub, new_image_filename + '_' + str(i)), z_slice)

        if do_no_norm:
            for i, z_slice in enumerate(image_data_no_norm):
                # save at new location (train or val)
                np.save(os.path.join(destination_folder_no_norm, sub, new_image_filename + '_' + str(i)), z_slice)