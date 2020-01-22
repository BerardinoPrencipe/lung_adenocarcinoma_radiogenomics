import os
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import torch
import medpy.metric.binary as mmb
from utils import normalize_data, perform_inference_volumetric_image
from projects.liver.train.config import window_hu

val_list = ["{:02d}".format(idx) for idx in range(1,3)]
val_list.append("10")

folder_dataset = 'datasets/ircadb'
folders_patients = os.listdir(folder_dataset)
folders_patients_train = [folder for folder in folders_patients if not any(val_el in folder for val_el in val_list)]
folders_patients_valid = [folder for folder in folders_patients if any(val_el in folder for val_el in val_list)]

folder_test_dataset = 'E:/Datasets/LiverScardapane'
folders_patients_test = os.listdir(folder_test_dataset)
folders_patients_test = [folder for folder in folders_patients_test
                         if os.path.isdir(os.path.join(folder_test_dataset, folder))]

path_net = 'logs/vessels/model_25D__2020-01-15__08_28_39.pht'
# Load net
net = torch.load(path_net)

# Start iteration over val set
for folder_patient_test in folders_patients_test:

    print('Processing ', folder_patient_test)
    path_test_pred = os.path.join(folder_test_dataset, folder_patient_test + ".nii")
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
    data = normalize_data(image_data, dmin=window_hu[0], dmax=window_hu[1])
    data = (data * 255).astype(np.uint8)

    # transpose so the z-axis (slices) are the first dimension
    data = np.transpose(data, (0, 2, 1))
    # CNN
    output = perform_inference_volumetric_image(net, data, context=2, do_round=True)
    output = np.transpose(output, (1, 2, 0)).astype(np.uint8)

    n_nonzero = np.count_nonzero(output)
    print("Non-zero elements = ", n_nonzero)

    output_nib_pre = nib.Nifti1Image(output, affine=None)
    nib.save(output_nib_pre, path_test_pred)
