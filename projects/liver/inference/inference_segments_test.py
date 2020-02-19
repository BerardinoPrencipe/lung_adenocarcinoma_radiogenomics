import os
import torch
import numpy as np
import SimpleITK as sitk

from utils_calc import normalize_data, normalize_data_old, perform_inference_volumetric_image, get_mcc
from projects.liver.train.config import window_hu
from semseg.models.vnet_v2 import VXNet

###################
### Loading CNN ###
###################
path_net = os.path.join('logs/segments/model_25D__2020-02-19__07_13_36.pht')
net = torch.load(path_net)

if torch.cuda.device_count() > 1:
    cuda_dev = torch.device('cuda:1')
else:
    cuda_dev = torch.device('cuda')
print('Device Count = {}, using CUDA Device = {}'.format(torch.cuda.device_count(), cuda_dev))

net = net.cuda(cuda_dev)
net.eval()

########################
### DICOM Image Test ###
########################
path_to_dicom_dataset = 'E:/Datasets/LiverScardapane'
# patient_name = '01_Dimastrochicco'
patient_name = '02_Lilla'
path_to_dicom_folder = os.path.join(path_to_dicom_dataset, patient_name)
reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(path_to_dicom_folder)
reader.SetFileNames(dicom_names)
image = reader.Execute()
spacing = image.GetSpacing()
data = sitk.GetArrayFromImage(image)

data = normalize_data(data, window_hu)

output = perform_inference_volumetric_image(net, data, context=2,
                                            do_round=False, do_argmax=True, cuda_dev=cuda_dev)

non_zero_elements = np.count_nonzero(output)
print("Non-Zero elements = {}".format(non_zero_elements))

output_sitk = sitk.GetImageFromArray(output)
output_sitk.SetSpacing(spacing=spacing)
path_pred = os.path.join(path_to_dicom_dataset, patient_name + '_segments.nii.gz')
sitk.WriteImage(output_sitk, path_pred)