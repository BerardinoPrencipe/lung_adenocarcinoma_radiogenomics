import os
import numpy as np
import torch
import nibabel as nib
import itk
import sys
import SimpleITK as sitk
import matplotlib.pyplot as plt
import medpy.metric.binary as mmb
import torch

current_path_abs = os.path.abspath('.')
sys.path.append(current_path_abs)
print('{} appended to sys!'.format(current_path_abs))

from projects.liver.train.config import use_local_path
from utils import normalize_data, perform_inference_volumetric_image, use_multi_gpu_model, \
    post_process_liver, get_dice, get_iou, get_patient_id
from projects.liver.train.config import window_hu
from semseg.models.vnet_v2 import VXNet

### variables ###
if use_local_path:
    test_folder = os.path.join(current_path_abs, 'datasets/Sliver_Nifti/Volumes')
    result_folder = os.path.join(current_path_abs, 'datasets/Sliver_Nifti/Results')
    gt_mask_folder = os.path.join(current_path_abs, 'datasets/Sliver_Nifti/GroundTruth')
else:
    test_folder = 'E:/Datasets/Sliver_Nifti/Volumes'
    result_folder = 'E:/Datasets/Sliver_Nifti/Results'
    gt_mask_folder = 'E:/Datasets/Sliver_Nifti/GroundTruth'

# alpha_beta = 'a5_b5'
alpha_beta = 'a3_b7'
method = 'CNN'
result_folder = os.path.join(result_folder, method, alpha_beta)

if not os.path.exists(result_folder):
    os.makedirs(result_folder)

results_folder_pre = os.path.join(result_folder, 'pre')
if not os.path.exists(results_folder_pre):
    os.makedirs(results_folder_pre)

results_folder_post = os.path.join(result_folder, 'post')
if not os.path.exists(results_folder_post):
    os.makedirs(results_folder_post)

#################

files_test_volumes = [file for file in os.listdir(test_folder)]

# load network
logs_dir = os.path.join(current_path_abs, 'logs/liver')
cuda = torch.cuda.is_available()
use_multi_gpu = True

# net_path = 'logs/liver/model_25D__2020-01-22__14_00_38.pht'
net_path = os.path.join(logs_dir, 'model_25D__2020-01-24__11_34_49.pht')
if use_local_path:
    # net_path = os.path.join(current_path_abs, logs_dir, 'model_epoch_0400.pht')
    net_path = os.path.join(logs_dir, max(os.listdir(logs_dir)))
print('Net Path = {}'.format(net_path))

if torch.cuda.device_count() > 1:
    cuda_dev = torch.device('cuda:1')
else:
    cuda_dev = torch.device('cuda')
print('Device Count = {}, using CUDA Device = {}'.format(torch.cuda.device_count(), cuda_dev))

use_state_dict = True

if use_state_dict:
    net = VXNet(dropout=True,context=2,num_outs=2)
    net.load_state_dict(torch.load(net_path))
else:
    net = torch.load(net_path)
net = net.cuda(cuda_dev)
net.eval()

eval_net_volumes = True

if eval_net_volumes:
    for idx, file_name_prediction in enumerate(files_test_volumes):
        print("Iter {} on {}".format(idx, len(files_test_volumes)-1))

        # load file
        data = nib.load(os.path.join(test_folder, file_name_prediction))

        # save affine
        input_aff = data.affine

        # convert to numpy
        data = data.get_data()

        # normalize data
        # data = normalize_data(data, dmin=window_hu[0], dmax=window_hu[1])
        data = normalize_data(data, window_hu)

        # transpose so the z-axis (slices) are the first dimension
        data = np.transpose(data, (2, 0, 1))

        output = perform_inference_volumetric_image(net, data, context=2, do_round=True, cuda_dev=cuda_dev)
        output = output.astype(np.uint8)

        # transpose so z-axis is last axis again and transform into nifti file
        output = np.transpose(output, (1, 2, 0)).astype(np.uint8)
        output_post = post_process_liver(output)

        output_nib_pre = nib.Nifti1Image(output, affine=input_aff)
        output_nib_post = nib.Nifti1Image(output_post, affine=input_aff)

        new_file_name = "segmentation-" + file_name_prediction.split("-")[-1]

        path_segmentation_pre = os.path.join(results_folder_pre, new_file_name)
        path_segmentation_post = os.path.join(results_folder_post, new_file_name)
        print("Path Pre  = ", path_segmentation_pre)
        print("Path Post = ", path_segmentation_post)
        nib.save(output_nib_pre, path_segmentation_pre)
        nib.save(output_nib_post, path_segmentation_post)


val_list = [idx for idx in range(1, 21)]

ious_pre   = np.zeros(len(val_list))
ious_post  = np.zeros(len(val_list))

dices_pre  = np.zeros(len(val_list))
dices_post = np.zeros(len(val_list))

rvds_pre   = np.zeros(len(val_list))
rvds_post  = np.zeros(len(val_list))


assds_pre  = np.zeros(len(val_list))
assds_post = np.zeros(len(val_list))

hds_pre    = np.zeros(len(val_list))
hds_post   = np.zeros(len(val_list))


paths_predictions_pre  = [filename for filename in os.listdir(results_folder_pre) if filename.endswith(".nii")]
paths_predictions_post = [filename for filename in os.listdir(results_folder_post) if filename.endswith(".nii")]

paths_ground_truth = [file for file in os.listdir(gt_mask_folder)]

paths_predictions_pre.sort()
paths_predictions_post.sort()
paths_ground_truth.sort()

eval_cnn = True

for p_id, (path_prediction_pre, path_prediction_post,
           path_ground_truth) in \
        enumerate(zip(paths_predictions_pre, paths_predictions_post,
                      paths_ground_truth)):

    print("Index {} on {}".format(p_id, len(paths_ground_truth)-1))

    prediction_mask_pre  = nib.load(os.path.join(results_folder_pre, path_prediction_pre))
    prediction_mask_pre  = prediction_mask_pre.get_data()
    prediction_mask_post = nib.load(os.path.join(results_folder_post, path_prediction_post))
    prediction_mask_post = prediction_mask_post.get_data()

    path_gt_mask = os.path.join(gt_mask_folder, path_ground_truth)

    ground_truth_mask = nib.load(path_gt_mask)
    voxel_spacing        = ground_truth_mask.header.get_zooms()
    ground_truth_mask    = ground_truth_mask.get_data()

    if eval_cnn:
        ious_pre[p_id]   = mmb.jc(prediction_mask_pre, ground_truth_mask)
        ious_post[p_id]  = mmb.jc(prediction_mask_post, ground_truth_mask)

        dices_pre[p_id]  = mmb.dc(prediction_mask_pre, ground_truth_mask)
        dices_post[p_id] = mmb.dc(prediction_mask_post, ground_truth_mask)

        rvds_pre[p_id]   = mmb.ravd(prediction_mask_pre, ground_truth_mask)
        rvds_post[p_id]  = mmb.ravd(prediction_mask_post, ground_truth_mask)

        # Average SSD
        assds_pre[p_id]  = mmb.assd(prediction_mask_pre, ground_truth_mask, voxelspacing=voxel_spacing)
        assds_post[p_id] = mmb.assd(prediction_mask_post, ground_truth_mask, voxelspacing=voxel_spacing)
        # Maximum SSD
        hds_pre[p_id]    = mmb.hd(prediction_mask_pre, ground_truth_mask, voxelspacing=voxel_spacing)
        hds_post[p_id]   = mmb.hd(prediction_mask_post, ground_truth_mask, voxelspacing=voxel_spacing)


avg_iou_pre  = np.mean(ious_pre)
avg_iou_post = np.mean(ious_post)

avg_dice_pre  = np.mean(dices_pre)
avg_dice_post = np.mean(dices_post)

avg_rvd_pre  = np.mean(rvds_pre)
avg_rvd_post = np.mean(rvds_post)

avg_assd_pre  = np.mean(assds_pre)
avg_assd_post = np.mean(assds_post)

avg_hd_pre  = np.mean(hds_pre)
avg_hd_post = np.mean(hds_post)

print("Average IoU  pre = {:.4f} post = {:.4f} ".format(avg_iou_pre, avg_iou_post))
print("Average Dice pre = {:.4f} post = {:.4f}".format(avg_dice_pre, avg_dice_post))
print("Average RVD  pre = {:+.3f} post = {:+.3f}".format(avg_rvd_pre, avg_rvd_post))
print("Average ASSD pre = {:.4f} post = {:.4f}".format(avg_assd_pre, avg_assd_post))
print("Average HD   pre = {:.4f} post = {:.4f}".format(avg_hd_pre, avg_hd_post))

