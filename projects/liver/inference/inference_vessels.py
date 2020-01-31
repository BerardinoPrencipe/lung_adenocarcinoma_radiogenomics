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

path_net_hv = 'logs/hv/model_25D__2020-01-16__16_42_37.pht'
path_net_pv = 'logs/pv/model_25D__2020-01-17__16_42_59.pht'
path_net = 'logs/vessels/model_25D__2020-01-15__08_28_39.pht'

# Load net
net = torch.load(path_net)

path_test_preds = list()

# Start iteration over val set
for idx, folder_patient_valid in enumerate(folders_patients_valid):

    path_test_image = os.path.join(folder_dataset, folder_patient_valid, 'image', 'image.nii')
    path_test_pred  = os.path.join(folder_dataset, folder_patient_valid, 'image', 'pred.nii')
    path_test_preds.append(path_test_pred)

    # load file
    data = nib.load(path_test_image)
    # save affine
    input_aff = data.affine
    # convert to numpy
    data = data.get_data()

    # normalize data
    # data = normalize_data(data, dmin=window_hu[0], dmax=window_hu[1])
    data = normalize_data(data, window_hu)

    # transpose so the z-axis (slices) are the first dimension
    data = np.transpose(data, (2, 0, 1))

    # CNN
    output = perform_inference_volumetric_image(net, data, context=2, do_round=True)
    output = np.transpose(output, (1, 2, 0)).astype(np.uint8)

    output_nib = nib.Nifti1Image(output, affine=input_aff)
    nib.save(output_nib, path_test_pred)
    print("Index {} on {}.\nImage saved in {}".format(idx, len(folders_patients_valid), path_test_pred))

ious       = np.zeros(len(val_list))
precisions = np.zeros(len(val_list))
recalls    = np.zeros(len(val_list))
dices      = np.zeros(len(val_list))
rvds       = np.zeros(len(val_list))
assds      = np.zeros(len(val_list))
hds        = np.zeros(len(val_list))

for idx, (folder_patient_valid, path_test_pred) in enumerate(zip(folders_patients_valid, path_test_preds)):

    # Ground Truth
    gt_pv_path = os.path.join(folder_dataset, folder_patient_valid, 'mask', 'pv.nii')
    gt_hv_path = os.path.join(folder_dataset, folder_patient_valid, 'mask', 'hv.nii')

    gt_pv_mask = nib.load(gt_pv_path)
    gt_pv_mask = gt_pv_mask.get_data()
    gt_hv_mask = nib.load(gt_hv_path)
    gt_hv_mask = gt_hv_mask.get_data()

    gt_vessels_mask = gt_pv_mask + gt_hv_mask

    voxel_spacing = gt_vessels_mask.header.get_zooms()

    # Output
    output = nib.load(path_test_pred)

    iou = mmb.jc(output, gt_vessels_mask)
    dice = mmb.dc(output, gt_vessels_mask)
    prec = mmb.precision(output, gt_vessels_mask)
    recall = mmb.recall(output, gt_vessels_mask)
    rvd = mmb.ravd(output, gt_vessels_mask)
    assd = mmb.assd(output, gt_vessels_mask, voxelspacing=voxel_spacing)
    hd = mmb.hd(output, gt_vessels_mask, voxelspacing=voxel_spacing)

    print('Patient   = ', folder_patient_valid)
    print('IoU       = ', iou)
    print('Dice      = ', dice)
    print('Precision = ', prec)
    print('Recall    = ', recall)
    print('RVD       = ', rvd)
    print('ASSD      = ', assd)
    print('HD        = ', hd)

avg_iou  = np.mean(ious)
avg_prec = np.mean(precisions)
avg_reca = np.mean(recalls)
avg_dice = np.mean(dices)
avg_rvd  = np.mean(rvds)
avg_assd = np.mean(assds)
avg_hd   = np.mean(hds)