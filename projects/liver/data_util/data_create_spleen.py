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
cross_val_steps = 4
trainval_images = 41

dataset_folder = os.path.join(current_path_abs, 'datasets/Task09_Spleen')
scan_folder = os.path.join(dataset_folder, 'imagesTr')
mask_folder = os.path.join(dataset_folder, 'labelsTr')
# source folder where the .nii.gz files are located

scan_filename = os.listdir(scan_folder)
scan_filename.sort()
print('Source Folder  = {}'.format(scan_folder))
print('Source SubDirs = {}'.format(scan_filename))

images_list = [filename.split("_")[1].split(".")[0] for filename in scan_filename]

# rand_perm_val = np.random.permutation(images_list)
rand_perm_val = ['008', '014', '010', '027', '029', '006', '017', '019', '063', '002',
                 '024', '044', '046', '041', '022', '016', '052', '033', '026', '049',
                 '056', '047', '021', '009', '018', '025', '060', '059', '032', '045',
                 '053', '040', '028', '031', '062', '061', '013', '038', '020', '012', '003']

print("Rand Perm Val = ", rand_perm_val)

# validation list
for idx_crossval in range(cross_val_steps):

    val_list = rand_perm_val[idx_crossval*(trainval_images//cross_val_steps):
                                  (idx_crossval+1)*(trainval_images//cross_val_steps)]
    if idx_crossval == cross_val_steps - 1:
        val_list.append(rand_perm_val[(len(rand_perm_val) - 1)])

    train_list = list(set(rand_perm_val)-set(val_list))

    print("Iter ", idx_crossval)
    print("Val   List = ", val_list)
    print("Train List = ", train_list)

    # destination folder where the subfolders with npy files will go
    destination_folder = os.path.join(dataset_folder, 'npy_crossval_{:03d}'.format(idx_crossval))
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

    for idx, filename in enumerate(scan_filename):

        image = nib.load(os.path.join(scan_folder, filename))
        mask = nib.load(os.path.join(mask_folder, filename))

        image = image.get_data()
        image = normalize_data(image, window_hu)
        image = np.transpose(image, (2, 0, 1))

        mask = mask.get_data()
        mask = np.transpose(mask, (2, 0, 1))

        new_image_filename = "volume-{}".format(idx)
        new_mask_filename = "segmentation-{}".format(idx)

        out_folder = "train" if filename.split("_")[1].split(".")[0] in train_list else "val"

        for i, slice in enumerate(image):
            np.save(os.path.join(destination_folder, out_folder, new_image_filename + '_' + str(i)), slice)

        for i, slice in enumerate(mask):
            np.save(os.path.join(destination_folder, out_folder, new_mask_filename + '_' + str(i)), slice)

    # for idx, subfolder_source in enumerate(source_subolders):
    #
    #     print('Index {} on {}'.format(idx, len(os.listdir(source_folder))-1))
    #     id_patient = subfolder_source[-2:]
    #     if id_patient not in images_list and id_patient not in val_list:
    #         print("Image is excluded")
    #         continue
    #     else:
    #         print("Image is not excluded")
    #
    #     image_folder = os.path.join(source_folder, subfolder_source, 'image')
    #     mask_folder = os.path.join(source_folder, subfolder_source, 'mask')
    #
    #     image_filename = os.path.join(image_folder, 'image.nii')
    #     mask_filename = os.path.join(mask_folder, 'mask.nii')
    #
    #     print("Image: ", image_filename)
    #     print("Mask : ", mask_filename)
    #
    #     # create new file name by stripping .nii and adding .npy
    #     new_image_filename = "volume-{}".format(id_patient)
    #     new_mask_filename = "segmentation-{}".format(id_patient)
    #
    #     # decide whether it will go to the train or val folder
    #     sub = subfolders[1] if id_patient in val_list else subfolders[0]
    #
    #     # load file
    #     image_data = nib.load(image_filename)
    #     mask_data = nib.load(mask_filename)
    #
    #     # convert to numpy
    #     mask_data = mask_data.get_data()
    #     mask_data_hv = (mask_data == VESSELS_CLASS[0]).astype(np.uint8)
    #     mask_data_pv = (mask_data == VESSELS_CLASS[1]).astype(np.uint8)
    #     mask_data          = np.logical_or(mask_data_hv, mask_data_pv)
    #     image_data_no_norm = image_data.get_data()
    #     image_data_norm    = normalize_data(image_data_no_norm, window_hu)
    #
    #     # transpose so the z-axis (slices) are the first dimension
    #     image_data_norm    = np.transpose(image_data_norm, (2,0,1))
    #     image_data_no_norm = np.transpose(image_data_no_norm, (2,0,1))
    #     mask_data          = np.transpose(mask_data, (2,0,1))
    #
    #     if do_mask_with_liver:
    #         # Load Liver Mask
    #         mask_liver_filename = os.path.join(mask_folder, 'liver_closed.nii')
    #         mask_liver_data = nib.load(mask_liver_filename)
    #         mask_liver_data = mask_liver_data.get_data()
    #         mask_liver_data = np.transpose(mask_liver_data, (2,0,1))
    #
    #         # Mask Image and Label with Liver Mask
    #         i_dtype = image_data_norm.dtype
    #         m_dtype = mask_data.dtype
    #         image_data_norm = (image_data_norm * mask_liver_data).astype(i_dtype)
    #         mask_data = (mask_data * mask_liver_data).astype(m_dtype)
    #
    #     print(f'[  Norm   ] Max = {image_data_norm.max()} - Min = {image_data_norm.min()}')
    #     print(f'[ No Norm ] Max = {image_data_no_norm.max()} - Min = {image_data_no_norm.min()}')
    #
    #     # loop through the mask slices
    #     for i, z_slice in enumerate(mask_data):
    #         # save at new location (train or val)
    #         np.save(os.path.join(destination_folder, sub, new_mask_filename + '_' + str(i)), z_slice)
    #
    #     # loop through the scan slices
    #     for i, z_slice in enumerate(image_data_norm):
    #         # save at new location (train or val)
    #         np.save(os.path.join(destination_folder, sub, new_image_filename + '_' + str(i)), z_slice)
