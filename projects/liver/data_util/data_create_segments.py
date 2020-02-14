import numpy as np
import nibabel as nib
import os
import sys
import platform

current_path_abs = os.path.abspath('.')
sys.path.append(current_path_abs)
print('{} appended to sys!'.format(current_path_abs))

from utils_calc import normalize_data
from utils import get_num_from_path
from projects.liver.train.config import window_hu

# Use local path or absolute
if 'Ubuntu' in platform.system() or 'Linux' in platform.system():
    isLinux = True
else:
    isLinux = False

############################
### SOURCE DATASET (nii) ###
############################
dataset_path_base = 'E:/Datasets/LiverDecathlon'
dataset_path_nii = os.path.join(dataset_path_base, 'nii')
source_images_folder = os.path.join(dataset_path_nii, 'images')
source_labels_segments_folder = os.path.join(dataset_path_nii, 'labels_segments')
source_labels_liver_folder = os.path.join(dataset_path_nii, 'labels_liver')

VAL_INT_NUM = 50

#################################
### DESTINATION DATASET (npy) ###
#################################
# create destination folder and possible subfolders
destination_folder = dataset_path_npy = os.path.join(dataset_path_base, 'npy')
destination_folder_masked = os.path.join(dataset_path_base, 'npy_masked')
subfolders = ["train", "val"]

if not os.path.isdir(destination_folder):
    os.makedirs(destination_folder)
    print('Created destionation_folder in {}'.format(destination_folder))

if not os.path.isdir(destination_folder_masked):
    os.makedirs(destination_folder_masked)
    print('Created destionation_folder_masked in {}'.format(destination_folder_masked))

for name in subfolders:
    if not os.path.isdir(os.path.join(destination_folder, name)):
        os.makedirs(os.path.join(destination_folder, name))
        print('Created destination        folder subdir: {}'.format(name))

    if not os.path.isdir(os.path.join(destination_folder_masked, name)):
        os.makedirs(os.path.join(destination_folder_masked, name))
        print('Created destination masked folder subdir: {}'.format(name))

print('Source Image  Folder = {}'.format(source_images_folder))
print('Source Labels Folder = {}'.format(source_labels_segments_folder))

# image_paths = os.listdir(source_images_folder)
# image_paths.sort()

label_paths = os.listdir(source_labels_segments_folder)
label_paths.sort()

for idx, label_path in enumerate(label_paths):

    print('Index {} on {}'.format(idx, len(label_paths)-1))

    image_filename = os.path.join(source_images_folder, label_path)
    label_filename = os.path.join(source_labels_segments_folder, label_path)
    liver_filename = os.path.join(source_labels_liver_folder, label_path)

    print("Image: ", image_filename)
    print("Label: ", label_filename)
    print("Liver: ", liver_filename)

    id_patient = get_num_from_path(label_path)
    # create new file name by stripping .nii and adding .npy
    new_image_filename = "volume-{}".format(id_patient)
    new_label_filename  = "segmentation-{}".format(id_patient)

    # decide whether it will go to the train or val folder
    sub = subfolders[1] if int(id_patient) < VAL_INT_NUM else subfolders[0]

    # load file
    image_data_nib = nib.load(image_filename)
    label_data_nib = nib.load(label_filename)
    liver_data_nib = nib.load(liver_filename)

    # convert to numpy
    image_data = image_data_nib.get_data()
    image_data = normalize_data(image_data, window_hu)

    label_data = label_data_nib.get_data()
    liver_data = liver_data_nib.get_data()

    print('Unique values of label data = {}'.format(np.unique(label_data)))
    print('Unique values of liver data = {}'.format(np.unique(liver_data)))

    # transpose so the z-axis (slices) are the first dimension
    image_data = np.transpose(image_data, (2, 0, 1))
    label_data = np.transpose(label_data, (2, 0, 1))
    liver_data = np.transpose(liver_data, (2, 0, 1))

    # mask label and image
    masked_label_data = liver_data * label_data
    masked_image_data = liver_data * image_data

    # loop through the mask slices
    for i, (z_slice, z_slice_masked) in enumerate(zip(label_data, masked_label_data)):
        # save at new location (train or val)
        np.save(os.path.join(destination_folder, sub, new_label_filename + '_' + str(i)), z_slice)
        np.save(os.path.join(destination_folder_masked, sub, new_label_filename + '_' + str(i)), z_slice_masked)

    # loop through the scan slices
    for i, (z_slice, z_slice_masked) in enumerate(zip(image_data, masked_image_data)):
        # save at new location (train or val)
        np.save(os.path.join(destination_folder, sub, new_image_filename + '_' + str(i)), z_slice)
        np.save(os.path.join(destination_folder_masked, sub, new_image_filename + '_' + str(i)), z_slice_masked)
