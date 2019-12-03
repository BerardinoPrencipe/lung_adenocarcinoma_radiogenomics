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

inference_on_test_set = False

# directory where to store nii.gz or numpy files
result_folder = 'E:/Datasets/LiTS/results'
if inference_on_test_set:
    result_folder = os.path.join(result_folder, 'test')
    test_folder = 'F:/Datasets/LiTS/test'
else:
    result_folder = os.path.join(result_folder, 'train')
    test_folder = 'F:/Datasets/LiTS/train'

if not os.path.exists(result_folder):
    os.makedirs(result_folder)

results_folder_pre = os.path.join(result_folder, 'pre')
if not os.path.exists(results_folder_pre):
    os.makedirs(results_folder_pre)

results_folder_post = os.path.join(result_folder, 'post')
if not os.path.exists(results_folder_post):
    os.makedirs(results_folder_post)

#################

if inference_on_test_set:
    # filter files that don't start with test
    files_test_volumes = [file for file in os.listdir(test_folder) if file[:4] == "test"]
else:
    files_test_volumes = [file for file in os.listdir(test_folder) if file[:6] == "volume"]

# load network
logs_dir = 'logs'
cuda = torch.cuda.is_available()
use_multi_gpu = True
if use_traced_model:
    if use_multi_gpu:
        path_traced_model = os.path.join(logs_dir, "traced_model_multi_" + model_name + ".pt")
    else:
        path_traced_model = os.path.join(logs_dir, "traced_model_" + model_name + ".pt")
    net = torch.jit.load(path_traced_model)
else:
    net = torch.load(os.path.join(logs_dir,"model_"+model_name+".pht"))
    if cuda and use_multi_gpu: net = use_multi_gpu_model(net)
    net.eval() # inference mode

for file_name in files_test_volumes:

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

    output_nib_pre = nib.Nifti1Image(output, affine=input_aff)
    output_nib_post = nib.Nifti1Image(output_post, affine=input_aff)

    if inference_on_test_set:
        new_file_name = "test-segmentation-" + file_name.split("-")[-1]
    else:
        new_file_name = "segmentation-" + file_name.split("-")[-1]

    path_segmentation_pre = os.path.join(results_folder_pre, new_file_name)
    path_segmentation_post = os.path.join(results_folder_post, new_file_name)
    print("Path Pre  = ", path_segmentation_pre)
    print("Path Post = ", path_segmentation_post)
    nib.save(output_nib_pre, path_segmentation_pre)
    nib.save(output_nib_post, path_segmentation_post)

# TODO: remove line below
# cpp_out = nib.load("F:/cpp_projects/__cpp_repos/example_app/out/build/x64-Release/segmentation-0.nii")
# data = nib.load("F:/Datasets/LiTS/train/volume-0.nii")