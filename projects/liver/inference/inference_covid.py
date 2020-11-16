import os
import configparser
import time

import SimpleITK as sitk
import nibabel as nib
import torch

from projects.liver.util.inference import perform_inference_volumetric_image
from utils import normalize_data

# Config
path_to_covid_dataset = 'H:\Datasets\CoVID-19\Scardapane_New'
window_hu = (-1000, 400)

# Net
# 2.5D V-Net
# path_net = 'logs/covid19/model_25D__2020-11-15__19_58_03.pht'
# path_net = 'logs/covid19/model_25D__2020-11-16__05_31_00.pht'
# context = 2

# 2D V-Net
context = 0
path_net = 'logs/covid19/model_25D__2020-11-16__13_06_15.pht'
print("Path Net = ", path_net)
net = torch.load(path_net)
cuda_device = torch.device('cuda:0')
net.to(cuda_device)
do_round = False
do_argmax = True

# Paths
subfolders = os.listdir(path_to_covid_dataset)
subfolders = [os.path.join(path_to_covid_dataset,s) for s in subfolders
              if os.path.isdir(os.path.join(path_to_covid_dataset,s))]

for idx, path_to_subfolder in enumerate(subfolders):
    subfiles = os.listdir(path_to_subfolder)
    paths_to_label = [s for s in subfiles if s.endswith('.nii.gz') and 'pred' not in s]
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

            print("Calculating output...")
            output_np = perform_inference_volumetric_image(net, image_normalized_np, context=context,
                                                          do_round=do_round, do_argmax=do_argmax,
                                                          cuda_dev=cuda_device)

            output_sitk = sitk.GetImageFromArray(output_np)
            output_sitk.SetOrigin(image_sitk.GetOrigin())
            output_sitk.SetDirection(image_sitk.GetDirection())
            output_sitk.SetSpacing(image_sitk.GetSpacing())

            print("Writing output...")
            out_filename = path_to_label[:-7]+"_pred.nii.gz"
            writer = sitk.ImageFileWriter()
            writer.SetImageIO("NiftiImageIO")
            writer.Execute(output_sitk, out_filename, True)

            reader = sitk.ImageFileReader()
            reader.SetImageIO("NiftiImageIO")
            reader.SetFileName(path_to_label)
            label_sitk = reader.Execute()
            label_size = label_sitk.GetSize()
            print("Label size :", label_size[0], label_size[1], label_size[2])
            label_np = sitk.GetArrayFromImage(label_sitk)
            print("Label shape: {}".format(label_np.shape))
