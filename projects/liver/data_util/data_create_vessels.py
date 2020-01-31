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
val_list = ["{:02d}".format(idx) for idx in range(1,5)]

dataset_folder = os.path.join(current_path_abs, 'datasets/ircadb')
# source folder where the .nii.gz files are located
source_folder = dataset_folder

#################

LIVER_CLASS = (1,)
VESSELS_CLASS = (2, 3)
ARTERY_CLASS = (4,)

# destination folder where the subfolders with npy files will go
if isLinux:
    destination_folder = os.path.join(current_path_abs, 'datasets/ircadb_npy')
else:
    destination_folder = 'E:/Datasets/ircadb'

# create destination folder and possible subfolders
subfolders = ["train", "val"]

if not os.path.isdir(destination_folder):
    os.makedirs(destination_folder)
    print('Created destionation_folder in {}'.format(destination_folder))

for name in subfolders:
    if not os.path.isdir(os.path.join(destination_folder, name)):
        os.makedirs(os.path.join(destination_folder, name))
        print('Created subdir: {}'.format(name))


source_subolders = os.listdir(source_folder).sort()
print('Source Folder  = {}'.format(source_folder))
print('Source SubDirs = {}'.format(source_subolders))

for idx, subfolder_source in enumerate(os.listdir(source_folder)):

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

    # decide whether it will go to the train or val folder
    sub = subfolders[1] if id_patient in val_list else subfolders[0]

    # load file
    image_data = nib.load(image_filename)
    mask_data = nib.load(mask_filename)

    # convert to numpy
    image_data = image_data.get_data()
    image_data = normalize_data(image_data, window_hu)
    mask_data = mask_data.get_data()

    # transpose so the z-axis (slices) are the first dimension
    image_data = np.transpose(image_data, (2,0,1))
    mask_data = np.transpose(mask_data, (2,0,1))

    # loop through the mask slices
    for i, z_slice in enumerate(mask_data):
        # save at new location (train or val)
        np.save(os.path.join(destination_folder, sub, new_mask_filename + '_' + str(i)), z_slice)

    # loop through the scan slices
    for i, z_slice in enumerate(image_data):
        # save at new location (train or val)
        np.save(os.path.join(destination_folder, sub, new_image_filename + '_' + str(i)), z_slice)
