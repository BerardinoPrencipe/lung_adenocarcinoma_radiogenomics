import os
import semseg.models.vnet
import numpy as np
import torch
import nibabel as nib
import itk
import SimpleITK as sitk
from utils import normalize_data, perform_inference_volumetric_image, use_multi_gpu_model, \
    post_process_liver, get_dice, get_iou, get_patient_id
from projects.liver.train.config import window_hu
import matplotlib.pyplot as plt
import medpy.metric.binary as mmb

### variables ###
test_folder = 'E:/Datasets/Sliver_Nifti/Volumes'
result_folder = 'E:/Datasets/Sliver_Nifti/Results'

alpha_beta = 'a5_b5'
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

sliver_scans_folder = 'E:/Datasets/Sliver_Nifti/Volumes'
files_test_volumes = [file for file in os.listdir(sliver_scans_folder)]

# load network
logs_dir = 'logs/liver'
cuda = torch.cuda.is_available()
use_multi_gpu = True

net_path = 'logs/liver/model_25D__2020-01-22__14_00_38.pht'
net = torch.load(net_path)
net = net.cuda()
net.eval()


for file_name_prediction in files_test_volumes:
    # load file
    data = nib.load(os.path.join(test_folder, file_name_prediction))

    # save affine
    input_aff = data.affine

    # convert to numpy
    data = data.get_data()

    # normalize data
    data = normalize_data(data, dmin=window_hu[0], dmax=window_hu[1])

    # transpose so the z-axis (slices) are the first dimension
    data = np.transpose(data, (2, 0, 1))

    output = perform_inference_volumetric_image(net, data, context=2, do_round=True)
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

ious_rg    = np.zeros(len(val_list))
dices_rg   = np.zeros(len(val_list))
rvds_rg    = np.zeros(len(val_list))
assds_rg   = np.zeros(len(val_list))
hds_rg     = np.zeros(len(val_list))

paths_predictions_pre  = [filename for filename in os.listdir(results_folder_pre) if filename.endswith(".nii")]
paths_predictions_post = [filename for filename in os.listdir(results_folder_post) if filename.endswith(".nii")]

sliver_masks_folder = 'E:/Datasets/Sliver_Nifti/GroundTruth'
paths_ground_truth = [file for file in os.listdir(sliver_masks_folder)]

region_growing_pred_folder = 'E:/Datasets/Sliver_Nifti/Results/RegionGrowing/D25'
paths_predictions_rg = [filename for filename in os.listdir(region_growing_pred_folder)]

paths_predictions_pre.sort()
paths_predictions_post.sort()
paths_ground_truth.sort()
paths_predictions_rg.sort()

eval_cnn = True
eval_rg  = False

for p_id, (path_prediction_pre, path_prediction_post, path_prediction_rg, path_ground_truth) in \
        enumerate(zip(paths_predictions_pre, paths_predictions_post, paths_predictions_rg, paths_ground_truth)):

    print("Index: ", p_id)

    prediction_mask_pre  = nib.load(os.path.join(results_folder_pre, path_prediction_pre))
    prediction_mask_pre  = prediction_mask_pre.get_data()
    prediction_mask_post = nib.load(os.path.join(results_folder_post, path_prediction_post))
    prediction_mask_post = prediction_mask_post.get_data()

    path_gt_mask = os.path.join(sliver_masks_folder, path_ground_truth)

    prediction_mask_rg = nib.load(os.path.join(region_growing_pred_folder, path_prediction_rg))
    prediction_mask_rg =  prediction_mask_rg.get_data()
    prediction_mask_rg[prediction_mask_rg == prediction_mask_rg.min()] = 0
    prediction_mask_rg[prediction_mask_rg == prediction_mask_rg.max()] = 1

    ground_truth_mask = nib.load(path_gt_mask)
    voxel_spacing        = ground_truth_mask.header.get_zooms()
    ground_truth_mask    = ground_truth_mask.get_data()


    if eval_rg:
        ious_rg[p_id]  = mmb.jc(prediction_mask_rg, ground_truth_mask)
        dices_rg[p_id] = mmb.dc(prediction_mask_rg, ground_truth_mask)
        rvds_rg[p_id]  = mmb.ravd(prediction_mask_rg, ground_truth_mask)
        assds_rg[p_id] = mmb.assd(prediction_mask_rg, ground_truth_mask)
        hds_rg[p_id]   = mmb.hd(prediction_mask_rg, ground_truth_mask)

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
avg_iou_rg   = np.mean(ious_rg)

avg_dice_pre  = np.mean(dices_pre)
avg_dice_post = np.mean(dices_post)
avg_dice_rg   = np.mean(dices_rg)

avg_rvd_pre  = np.mean(rvds_pre)
avg_rvd_post = np.mean(rvds_post)
avg_rvd_rg   = np.mean(rvds_rg)

avg_assd_pre  = np.mean(assds_pre)
avg_assd_post = np.mean(assds_post)
avg_assd_rg   = np.mean(assds_rg)

avg_hd_pre  = np.mean(hds_pre)
avg_hd_post = np.mean(hds_post)
avg_hd_rg   = np.mean(hds_rg)

print("Average IoU  pre = {:.4f} post = {:.4f} rg = {:.4f}".format(avg_iou_pre, avg_iou_post, avg_iou_rg))
print("Average Dice pre = {:.4f} post = {:.4f} rg = {:.4f}".format(avg_dice_pre, avg_dice_post, avg_dice_rg))
print("Average RVD  pre = {:+.3f} post = {:+.3f} rg = {:+.3f}".format(avg_rvd_pre, avg_rvd_post, avg_rvd_rg))
print("Average ASSD pre = {:.4f} post = {:.4f} rg = {:.4f}".format(avg_assd_pre, avg_assd_post, avg_assd_rg))
print("Average HD   pre = {:.4f} post = {:.4f} rg = {:.4f}".format(avg_hd_pre, avg_hd_post, avg_hd_rg))

# TODO: remove line below
# cpp_out = nib.load("F:/cpp_projects/__cpp_repos/example_app/out/build/x64-Release/segmentation-0.nii")
# data = nib.load("F:/Datasets/LiTS/train/volume-0.nii")