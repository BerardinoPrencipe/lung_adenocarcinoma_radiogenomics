import os
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import torch
import medpy.metric.binary as mmb
import sys

current_path_abs = os.path.abspath('.')
sys.path.append(current_path_abs)
print('{} appended to sys!'.format(current_path_abs))

from utils_calc import normalize_data, normalize_data_old, perform_inference_volumetric_image
from projects.liver.train.config import window_hu
from semseg.models.vnet_v2 import VXNet

# NEW VERSION
val_list = ["{:02d}".format(idx) for idx in range(1,5)]
print('Val List = {}'.format(val_list))

# OLD VERSION
# val_list = ["{:02d}".format(idx) for idx in range(1,3)]
# val_list.append("10")

folder_dataset = os.path.join(current_path_abs, 'datasets/ircadb')
print('Dataset Folder = {}'.format(folder_dataset))
folders_patients = os.listdir(folder_dataset)
folders_patients_train = [folder for folder in folders_patients if not any(val_el in folder for val_el in val_list)]
folders_patients_valid = [folder for folder in folders_patients if any(val_el in folder for val_el in val_list)]
folders_patients_valid.sort()

# OLD VERSION
# path_net = os.path.join(current_path_abs, 'logs/vessels/model_25D__2020-01-15__08_28_39.pht')

# NEW VERSION
use_state_dict = True

# Load net
if use_state_dict:
    path_net = os.path.join(current_path_abs, 'logs/vessels/model_epoch_0400.pht')
    net = VXNet(dropout=True,context=2,num_outs=2)
    net.load_state_dict(torch.load(path_net))
else:
    path_net = os.path.join(current_path_abs, 'logs/vessels/model_25D__2020-02-01__16_42_49.pht')
    net = torch.load(path_net)

if torch.cuda.device_count() > 1:
    cuda_dev = torch.device('cuda:1')
else:
    cuda_dev = torch.device('cuda')
print('Device Count = {}, using CUDA Device = {}'.format(torch.cuda.device_count(), cuda_dev))

net = net.cuda(cuda_dev)
net.eval()

print('Network Path = {}'.format(path_net))

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
    # RIGHT AND HEALTHY VERSION
    data = normalize_data(data, window_hu)

    # WRONG AND UGLY VERSION
    # data = normalize_data_old(data, dmin=window_hu[0], dmax=window_hu[1])
    # data = (data * 255).astype(np.uint8)

    # transpose so the z-axis (slices) are the first dimension
    data = np.transpose(data, (2, 0, 1))

    # CNN
    output = perform_inference_volumetric_image(net, data, context=2, do_round=True, cuda_dev=cuda_dev)
    output = np.transpose(output, (1, 2, 0)).astype(np.uint8)

    non_zero_elements = np.count_nonzero(output)
    print("Non-Zero elements = {}".format(non_zero_elements))

    output_nib = nib.Nifti1Image(output, affine=input_aff)
    nib.save(output_nib, path_test_pred)
    print("Index {} on {}.\nImage saved in {}".format(idx, len(folders_patients_valid)-1, path_test_pred))

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
    voxel_spacing = gt_pv_mask.header.get_zooms()
    gt_pv_mask = gt_pv_mask.get_data()
    gt_hv_mask = nib.load(gt_hv_path)
    gt_hv_mask = gt_hv_mask.get_data()

    gt_vessels_mask = np.logical_or(gt_pv_mask, gt_hv_mask)

    # Output
    output = nib.load(path_test_pred)
    output = output.get_data()

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

    ious[idx] = iou
    precisions[idx] = prec
    recalls[idx] = recall
    dices[idx] = dice
    rvds[idx] = rvd
    assds[idx] = assd
    hds[idx] = hd

avg_iou  = np.mean(ious)
avg_prec = np.mean(precisions)
avg_reca = np.mean(recalls)
avg_dice = np.mean(dices)
avg_rvd  = np.mean(rvds)
avg_assd = np.mean(assds)
avg_hd   = np.mean(hds)

print('Average IoU       = {}'.format(avg_iou))
print('Average Precision = {}'.format(avg_prec))
print('Average Recall    = {}'.format(avg_reca))
print('Average Dice      = {}'.format(avg_dice))
print('Average RVD       = {}'.format(avg_rvd))
print('Average ASSD      = {}'.format(avg_assd))
print('Average MSSD      = {}'.format(avg_hd))

import json

data = {
    'IoU'       : avg_iou,
    'Precision' : avg_prec,
    'Recall'    : avg_reca,
    'Dice'      : avg_dice,
    'RVD'       : avg_rvd,
    'ASSD'      : avg_assd,
    'MSSD'      : avg_hd,
}

json_path = os.path.join(current_path_abs, 'datasets/ircadb_metrics.json')
print('JSON Path = {}'.format(json_path))

with open(json_path, 'w') as f:
    json.dump(data, f)