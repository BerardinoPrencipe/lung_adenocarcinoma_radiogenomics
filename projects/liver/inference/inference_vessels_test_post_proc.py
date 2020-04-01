import os
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import torch
import medpy.metric.binary as mmb
from utils_calc import normalize_data
from projects.liver.util.inference import perform_inference_volumetric_image
from projects.liver.train.config import window_hu

use_in_lab = False
if use_in_lab:
    folder_test_dataset = 'E:/Datasets/LiverScardapane'
else:
    folder_test_dataset = 'H:/Datasets/Liver/LiverScardapane'
folder_test_images = os.path.join(folder_test_dataset, 'ct_scans')
folders_patients_test = os.listdir(folder_test_images)
folders_patients_test = [folder for folder in folders_patients_test
                         if os.path.isdir(os.path.join(folder_test_images, folder))]

path_net = None
do_round = True
do_argmax = False
model_to_use = "s_multi"
folder_test_pred = os.path.join(folder_test_dataset, model_to_use)
if not os.path.exists(folder_test_pred):
    os.makedirs(folder_test_pred)

if model_to_use == "ircadb":
    # path_net = 'logs/vessels/model_25D__2020-01-15__08_28_39.pht'
    path_net = 'logs/vessels/model_25D__2020-03-12__10_37_59.pht'
elif model_to_use == "s_multi":
    path_net = 'logs/vessels_scardapane/model_25D__2020-03-27__07_11_38.pht'
    do_round = False
    do_argmax = True
elif model_to_use == "s_single":
    path_net = 'logs/vessels_scardapane_one_class/model_25D__2020-03-28__04_43_26.pht'

path_net = 'logs/vessels/model_25D__2020-03-12__10_37_59.pht'
# Load net
net = torch.load(path_net)
cuda_device = torch.device('cuda:0')
net.to(cuda_device)

# Start iteration over val set
for idx, folder_patient_test in enumerate(folders_patients_test[:1]):
    print('Starting iter {} on {}'.format(idx+1,len(folders_patients_test)))
    print('Processing ', folder_patient_test)
    path_test_pred = os.path.join(folder_test_pred, folder_patient_test + ".nii.gz")
    path_test_folder = os.path.join(folder_test_images, folder_patient_test)

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path_test_folder)
    reader.SetFileNames(dicom_names)
    image_sitk = reader.Execute()
    size = image_sitk.GetSize()
    print("Image size: ", size[0], size[1], size[2])
    spacing = image_sitk.GetSpacing()
    image_np = sitk.GetArrayFromImage(image_sitk)

    # normalize data
    data = normalize_data(image_np, window_hu)

    # transpose so the z-axis (slices) are the first dimension
    data = np.transpose(data, (0, 2, 1))
    # CNN
    output_np = perform_inference_volumetric_image(net, data, context=2,
                                                   do_round=do_round, do_argmax=do_argmax,
                                                   cuda_dev=cuda_device)
    output_np = np.transpose(output_np, (1, 2, 0)).astype(np.uint8)

    n_nonzero = np.count_nonzero(output_np)
    print("Non-zero elements = ", n_nonzero)

    output_nib_pre = nib.Nifti1Image(output_np, affine=None)
    nib.save(output_nib_pre, path_test_pred)


# Post processing
# output_sitk = sitk.GetImageFromArray(output_np)
# seed = np.where(output_np)
# seg_conn_sitk = sitk.ConnectedThreshold(image_sitk, seedList=[seed],
#                                         lower=100, upper=190)