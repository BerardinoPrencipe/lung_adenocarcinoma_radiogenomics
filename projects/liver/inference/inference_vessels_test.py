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
folders_patients_test = os.listdir(folder_test_dataset)
folders_patients_test = [folder for folder in folders_patients_test
                         if os.path.isdir(os.path.join(folder_test_dataset, folder))]

# path_net = 'logs/vessels/model_25D__2020-01-15__08_28_39.pht'
# path_net = 'logs/vessels/model_25D__2020-03-12__10_37_59.pht'
path_net = 'logs/vessels_scardapane/model_25D__2020-03-27__07_11_38.pht'
# Load net
net = torch.load(path_net)
cuda_device = torch.device('cuda:0')
net.to(cuda_device)

# Start iteration over val set
for idx, folder_patient_test in enumerate(folders_patients_test):
    print('Starting iter {} on {}'.format(idx+1,len(folders_patients_test)))
    print('Processing ', folder_patient_test)
    path_test_pred = os.path.join(folder_test_dataset, folder_patient_test + ".nii.gz")
    path_test_folder = os.path.join(folder_test_dataset, folder_patient_test)

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path_test_folder)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    size = image.GetSize()
    print("Image size: ", size[0], size[1], size[2])
    spacing = image.GetSpacing()
    image_data = sitk.GetArrayFromImage(image)

    # normalize data
    # data = normalize_data(image_data, dmin=window_hu[0], dmax=window_hu[1])
    data = normalize_data(image_data, window_hu)
    data = (data * 255).astype(np.uint8)

    # transpose so the z-axis (slices) are the first dimension
    data = np.transpose(data, (0, 2, 1))
    # CNN
    output = perform_inference_volumetric_image(net, data, context=2, do_round=True, cuda_dev=cuda_device)
    output = np.transpose(output, (1, 2, 0)).astype(np.uint8)

    n_nonzero = np.count_nonzero(output)
    print("Non-zero elements = ", n_nonzero)

    output_nib_pre = nib.Nifti1Image(output, affine=None)
    nib.save(output_nib_pre, path_test_pred)
