import numpy as np
import nibabel as nib
import os.path
from utils import normalize_data, get_patient_id
from projects.liver.train.config import window_hu, use_local_path

### variables ###

# validation list
val_list = [idx for idx in range(20)]

if use_local_path:
    dataset_folder = 'datasets/LiTS/train'
else:
    dataset_folder = 'F:/Datasets/LiTS/train'

# source folder where the .nii.gz files are located
source_folder = dataset_folder

#################

LIVER_CLASS = 1
TUMOR_CLASS = 2

# destination folder where the subfolders with npy files will go
if use_local_path:
    destination_folder = 'datasets/LiTS/npy'
else:
    destination_folder = 'E:/Datasets/LiTS'


# create destination folder and possible subfolders
subfolders = ["train", "val"]
if not os.path.isdir(destination_folder):
	os.makedirs(destination_folder)
for name in subfolders:
    if not os.path.isdir(os.path.join(destination_folder, name)):
        os.makedirs(os.path.join(destination_folder, name))

for file_name in os.listdir(source_folder):

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
        # data = normalize_data_old(data, dmin=window_hu[0], dmax=window_hu[1])
        data = normalize_data(data, window_hu)
        print('[After  Normalization] data.min() = {} data.max() = {}'.format(data.min(),data.max()))

    # check if it is a segmentation file and select only the tumor (2) as positive label
    if file_name[:3] == 'seg': data = (data==LIVER_CLASS).astype(np.uint8)

    # transpose so the z-axis (slices) are the first dimension
    data = np.transpose(data, (2, 0, 1))

    # loop through the slices
    for i, z_slice in enumerate(data):

        # save at new location (train or val)
        np.save(os.path.join(destination_folder, sub, new_file_name + '_' + str(i)), z_slice)
