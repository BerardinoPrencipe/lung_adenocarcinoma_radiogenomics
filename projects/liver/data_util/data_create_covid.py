import os
import configparser
import numpy as np
import random
import SimpleITK as sitk
from tqdm import tqdm
from utils import normalize_data

# Config
# Home PC
path_to_covid_dataset = 'H:\Datasets\CoVID-19\Scardapane_New'
path_to_covid_slices  = os.path.join('datasets', 'CoVID19', 'npy')
path_to_train = os.path.join(path_to_covid_slices, 'train')
path_to_val = os.path.join(path_to_covid_slices, 'val')
paths_trainval = [path_to_train, path_to_val]
train_percentage = 0.8
window_hu = (-1000, 400)

# Script
os.makedirs(path_to_covid_slices, exist_ok=True)
for path_ in paths_trainval:
    os.makedirs(path_, exist_ok=True)
subfolders = os.listdir(path_to_covid_dataset)
subfolders = [os.path.join(path_to_covid_dataset,s) for s in subfolders
              if os.path.isdir(os.path.join(path_to_covid_dataset,s))]

for idx, path_to_subfolder in enumerate(subfolders):
    subfiles = os.listdir(path_to_subfolder)
    paths_to_label = [s for s in subfiles if s.endswith('.nii.gz')]
    paths_to_label = [os.path.join(path_to_subfolder, s) for s in paths_to_label]
    paths_to_series_dir = [s[:-7] for s in paths_to_label]
    paths_to_ini = [s+".ini" for s in paths_to_series_dir]

    print("Reading Image [{:2d}/{:2d}]: {}".format(idx+1, len(subfolders), paths_to_series_dir))
    for path_to_label, path_to_series_dir, path_to_ini \
            in zip(paths_to_label, paths_to_series_dir, paths_to_ini):

        config = configparser.ConfigParser()
        config.read(path_to_ini)
        ss = config.sections()
        print("ini sections = {}".format(ss))
        if 'annotation' in ss:
            start_slice = int(config['annotation']['StartSlice'])
            end_slice   = int(config['annotation']['EndSlice'])
            step        = int(config['annotation']['Step'])
            print("[Config] Slices: [{} - {}], Step: {}".format(start_slice, end_slice, step))
            slices = range(start_slice, end_slice+step, step)
            print("[Range ] Slices: [{} - {}], Step: {}".format(slices[0], slices[-1], step))

            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(path_to_series_dir)
            reader.SetFileNames(dicom_names)

            image_sitk = reader.Execute()
            image_size = image_sitk.GetSize()
            print("Image size :", image_size[0], image_size[1], image_size[2])
            image_np = sitk.GetArrayFromImage(image_sitk)
            print("Image shape: {}".format(image_np.shape))
            image_normalized_np = normalize_data(image_np, window_hu)

            reader = sitk.ImageFileReader()
            reader.SetImageIO("NiftiImageIO")
            reader.SetFileName(path_to_label)
            label_sitk = reader.Execute()
            label_size = label_sitk.GetSize()
            print("Label size :", label_size[0], label_size[1], label_size[2])
            label_np = sitk.GetArrayFromImage(label_sitk)
            print("Label shape: {}".format(label_np.shape))

            pop_size = len(slices)
            train_size = int(train_percentage * pop_size)
            val_size = pop_size - train_size
            print("Train Size = {:3d}".format(train_size))
            print("Val   Size = {:3d}".format(val_size))
            print("Pop   Size = {:3d}".format(pop_size))
            slices_train = random.sample(slices, train_size)

            id_patient = os.path.split(path_to_series_dir)[-1]
            new_image_filename = "volume-{}".format(id_patient)
            new_label_filename = "segmentation-{}".format(id_patient)
            cnt_no_1 = 0
            cnt_no_2 = 0
            for i in tqdm(slices):
                z_slice_label = label_np[i]
                z_slice_image = image_normalized_np[i]

                # Replace healthy parenchima (4) with background (0)
                z_slice_label[z_slice_label==4] = 0
                # Replace consolidation (3) with linear opacity (2)
                z_slice_label[z_slice_label==3] = 2

                assert (z_slice_label > 2).sum() + (z_slice_label < 0).sum() == 0, "Uncorrect Label! It must be in {0,1,2}"
                assert (z_slice_image > 1).sum() + (z_slice_image < 0).sum() == 0, "Uncorrect normalization! It must be in [0,1]"

                if (z_slice_label == 1).sum() == 0:
                    cnt_no_1 += 1
                if (z_slice_label == 2).sum() == 0:
                    cnt_no_2 += 1

                if i in slices_train:
                    np.save(os.path.join(path_to_train, new_label_filename + '_' + str(i)), z_slice_label)
                    np.save(os.path.join(path_to_train, new_image_filename + '_' + str(i)), z_slice_image)
                else:
                    np.save(os.path.join(path_to_val, new_label_filename + '_' + str(i)), z_slice_label)
                    np.save(os.path.join(path_to_val, new_image_filename + '_' + str(i)), z_slice_image)
            print("Counter No Label 1 = {}".format(cnt_no_1))
            print("Counter No Label 2 = {}".format(cnt_no_2))
            print("Slices             = {}".format(len(slices)))