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
do_only_final_cross_val = True
do_mask_with_liver = True
cross_val_steps = 4
trainval_images = 16
test_img = "02"
exclude_list = [test_img, "10","12","20"]
# val_list = ["{:02d}".format(idx) for idx in range(1,3)]
# exclude_list = [] # exclude_list = ["10"]

dataset_folder = os.path.join(current_path_abs, 'datasets/ircadb')
# source folder where the .nii.gz files are located
source_folder = os.path.join(dataset_folder, 'nii')

source_subolders = os.listdir(source_folder)
source_subolders.sort()
print('Source Folder  = {}'.format(source_folder))
print('Source SubDirs = {}'.format(source_subolders))

LIVER_CLASS = (1,)
VESSELS_CLASS = (2, 3)
ARTERY_CLASS = (4,)

# rand_perm_val = np.random.permutation(trainval_images)
rand_perm_val = [ 7, 12,  8, 11,
                  1,  4, 10,  9,
                  5,  0,  3, 13,
                 14,  6,  2, 15]
print("Rand Perm Val = ", rand_perm_val)

# validation list
for idx_crossval in range(cross_val_steps+1):
    if idx_crossval == cross_val_steps:
        val_list = [test_img,]
    else:
        if do_only_final_cross_val:
            continue
        rand_perm_val_list = ["{:02d}".format(idx+1) for idx in rand_perm_val]
        val_list = rand_perm_val_list[idx_crossval*(trainval_images//cross_val_steps):
                                      (idx_crossval+1)*(trainval_images//cross_val_steps)]
    print("Iter ", idx_crossval)
    print("Val List = ", val_list)

    # destination folder where the subfolders with npy files will go
    destination_folder = os.path.join(dataset_folder, 'npy_crossval_{:02d}'.format(idx_crossval))
    print("Destination Folder = ", destination_folder)

    # create destination folder and possible subfolders
    subfolders = ["train", "val"]

    if not os.path.isdir(destination_folder):
        os.makedirs(destination_folder)
        print('Created destionation_folder in {}'.format(destination_folder))

    for name in subfolders:
        if not os.path.isdir(os.path.join(destination_folder, name)):
            os.makedirs(os.path.join(destination_folder, name))
            print('Created subdir: {}'.format(name))

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

        if id_patient in exclude_list and id_patient not in val_list:
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
            image_data_no_norm = np.transpose(image_data_no_norm, (2,0,1))
            mask_data          = np.transpose(mask_data, (2,0,1))

            if do_mask_with_liver:
                # Load Liver Mask
                mask_liver_filename = os.path.join(mask_folder, 'liver_closed.nii')
                mask_liver_data = nib.load(mask_liver_filename)
                mask_liver_data = mask_liver_data.get_data()
                mask_liver_data = np.transpose(mask_liver_data, (2,0,1))

                # Mask Image and Label with Liver Mask
                i_dtype = image_data_norm.dtype
                m_dtype = mask_data.dtype
                image_data_norm = (image_data_norm * mask_liver_data).astype(i_dtype)
                mask_data = (mask_data * mask_liver_data).astype(m_dtype)

            print(f'[  Norm   ] Max = {image_data_norm.max()} - Min = {image_data_norm.min()}')
            print(f'[ No Norm ] Max = {image_data_no_norm.max()} - Min = {image_data_no_norm.min()}')

            # loop through the mask slices
            for i, z_slice in enumerate(mask_data):
                # save at new location (train or val)
                np.save(os.path.join(destination_folder, sub, new_mask_filename + '_' + str(i)), z_slice)

            # loop through the scan slices
            for i, z_slice in enumerate(image_data_norm):
                # save at new location (train or val)
                np.save(os.path.join(destination_folder, sub, new_image_filename + '_' + str(i)), z_slice)
