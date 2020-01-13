import numpy as np
import nibabel as nib
import os.path
from utils import normalize_data, get_patient_id

### variables ###

# validation list
val_list = ["{:02d}".format(idx) for idx in range(3)]
val_list.append("10")

dataset_folder = 'datasets/ircadb'
# source folder where the .nii.gz files are located
source_folder = dataset_folder

#################

LIVER_CLASS = (1,)
VESSELS_CLASS = (2, 3)
ARTERY_CLASS = (4,)

# destination folder where the subfolders with npy files will go
destination_folder = 'E:/Datasets/ircadb'

# create destination folder and possible subfolders
subfolders = ["train", "val"]
if not os.path.isdir(destination_folder):
	os.makedirs(destination_folder)
for name in subfolders:
    if not os.path.isdir(os.path.join(destination_folder, name)):
        os.makedirs(os.path.join(destination_folder, name))

for subfolder_source in os.listdir(source_folder):

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
    mask_data = mask_data.get_data()

    mask_data_hv = (mask_data == VESSELS_CLASS[0]).astype(np.uint8)
    mask_data_pv = (mask_data == VESSELS_CLASS[1]).astype(np.uint8)
    mask_data = np.logical_or(mask_data_hv, mask_data_pv)

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
