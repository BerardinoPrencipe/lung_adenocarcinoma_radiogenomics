import os
import networks
import numpy as np
import torch
import nibabel as nib
import itk
import SimpleITK as sitk
from utils import normalize_data, perform_inference_volumetric_image, use_multi_gpu_model, \
                  post_process_liver, get_dice, get_iou, get_patient_id
import matplotlib.pyplot as plt
import medpy.metric.binary as mmb

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

for file_name_prediction in files_test_volumes[:1]:

    # load file
    data = nib.load(os.path.join(test_folder, file_name_prediction))

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
        new_file_name = "test-segmentation-" + file_name_prediction.split("-")[-1]
    else:
        new_file_name = "segmentation-" + file_name_prediction.split("-")[-1]

    path_segmentation_pre = os.path.join(results_folder_pre, new_file_name)
    path_segmentation_post = os.path.join(results_folder_post, new_file_name)
    print("Path Pre  = ", path_segmentation_pre)
    print("Path Post = ", path_segmentation_post)
    nib.save(output_nib_pre, path_segmentation_pre)
    nib.save(output_nib_post, path_segmentation_post)


if inference_on_test_set:
    val_list = [idx for idx in range(len(files_test_volumes))]
else:
    val_list = [idx for idx in range(20)]

ious_pre = np.zeros(len(val_list))
ious_post = np.zeros(len(val_list))
dices_pre = np.zeros(len(val_list))
dices_post = np.zeros(len(val_list))
rvds_pre = np.zeros(len(val_list))
rvds_post = np.zeros(len(val_list))

paths_predictions_pre  = [filename for filename in os.listdir(results_folder_pre) if filename.endswith(".nii")]
paths_predictions_post = [filename for filename in os.listdir(results_folder_post) if filename.endswith(".nii")]
paths_ground_truth     = [filename for filename in os.listdir(test_folder) if filename[:12] == "segmentation"]

paths_predictions_pre.sort()
paths_predictions_post.sort()
paths_ground_truth.sort()

for idx, (path_prediction_pre, path_prediction_post, path_ground_truth) in \
        enumerate(zip(paths_predictions_pre, paths_predictions_post, paths_ground_truth)):

    print("Index: ", idx)
    p_id = get_patient_id(path_ground_truth)
    if p_id not in val_list:
        continue
    else:
        print("Patient ID = ", p_id)

    prediction_mask_pre  = nib.load(os.path.join(results_folder_pre, path_prediction_pre))
    prediction_mask_pre  = prediction_mask_pre.get_data()
    prediction_mask_post = nib.load(os.path.join(results_folder_post, path_prediction_post))
    prediction_mask_post = prediction_mask_post.get_data()
    ground_truth_mask    = nib.load(os.path.join(test_folder, path_ground_truth))
    ground_truth_mask    = ground_truth_mask.get_data()

    ious_pre[p_id]   = mmb.jc(prediction_mask_pre, ground_truth_mask)
    ious_post[p_id]  = mmb.jc(prediction_mask_post, ground_truth_mask)

    dices_pre[p_id]  = mmb.dc(prediction_mask_pre, ground_truth_mask)
    dices_post[p_id] = mmb.dc(prediction_mask_post, ground_truth_mask)

    rvds_pre[p_id]   = mmb.ravd(prediction_mask_pre, ground_truth_mask)
    rvds_post[p_id]  = mmb.ravd(prediction_mask_post, ground_truth_mask)

avg_iou_pre  = np.mean(ious_pre)
avg_iou_post = np.mean(ious_post)

avg_dice_pre  = np.mean(dices_pre)
avg_dice_post = np.mean(dices_post)

avg_rvd_pre  = np.mean(rvds_pre)
avg_rvd_post = np.mean(rvds_post)

print("Average IoU  pre = {:.4f} post = {:.4f}".format(avg_iou_pre, avg_iou_post))
print("Average Dice pre = {:.4f} post = {:.4f}".format(avg_dice_pre, avg_dice_post))
print("Average RVD  pre = {:+.3f} post = {:+.3f}".format(avg_rvd_pre, avg_rvd_post))

# TODO: remove line below
# cpp_out = nib.load("F:/cpp_projects/__cpp_repos/example_app/out/build/x64-Release/segmentation-0.nii")
# data = nib.load("F:/Datasets/LiTS/train/volume-0.nii")