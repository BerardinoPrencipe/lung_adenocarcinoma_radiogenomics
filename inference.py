import os
import networks
import numpy as np
import torch
import nibabel as nib
import itk
import SimpleITK as sitk
from utils import normalize_data, perform_inference_volumetric_image, use_multi_gpu_model, post_process_liver
import matplotlib.pyplot as plt

### variables ###

# name of the model saved
model_name = '25D'
use_traced_model = True

# the number of context slices before and after as defined as in train.py before training
context = 2

LIVER_CLASS = 1
TUMOR_CLASS = 2

# directory where to store nii.gz or numpy files
result_folder = 'E:/Datasets/LiTS/results'
test_folder = 'F:/Datasets/LiTS/test'

#################

# create result folder if neccessary
if not os.path.isdir(result_folder):
	os.makedirs(result_folder)

# filter files that don't start with test
files = [file for file in os.listdir(test_folder) if file[:4]=="test"]

# load network
logs_dir = 'logs'
cuda = torch.cuda.is_available()
use_multi_gpu = False
if use_traced_model:
    net = torch.jit.load(os.path.join(logs_dir, "traced_model_" + model_name + ".pt"))
else:
    net = torch.load(os.path.join(logs_dir,"model_"+model_name+".pht"))
    if cuda and use_multi_gpu: net = use_multi_gpu_model(net)
    net.eval() # inference mode

for file_name in files[:1]:

    # load file
    data = nib.load(os.path.join(test_folder, file_name))

    # save affine
    input_aff = data.affine

    # convert to numpy
    data = data.get_data()

    # normalize data
    data = normalize_data(data, dmin=-200, dmax=200)

    # transpose so the z-axis (slices) are the first dimension
    data = np.transpose(data, (2, 0, 1))

    output = perform_inference_volumetric_image(net, data, context=2, do_round=True)
    output = output.astype(np.uint8)

    # transpose so z-axis is last axis again and transform into nifti file
    output = np.transpose(output, (1, 2, 0)).astype(np.uint8)
    output_post = post_process_liver(output)

    output_nib = nib.Nifti1Image(output, affine=input_aff)
    output_nib_after = nib.Nifti1Image(output_post, affine=input_aff)

    new_file_name = "test-segmentation-" + file_name.split("-")[-1]
    new_file_name_after = "test-segmentation-after-" + file_name.split("-")[-1]
    print(new_file_name)
    print(new_file_name_after)

    nib.save(output_nib, os.path.join(result_folder, new_file_name))
    nib.save(output_nib_after, os.path.join(result_folder, new_file_name_after))
