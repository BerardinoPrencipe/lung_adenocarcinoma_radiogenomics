import os
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import torch
import medpy.metric.binary as mmb
import sys
from sklearn.metrics import confusion_matrix, matthews_corrcoef

current_path_abs = os.path.abspath('.')
sys.path.append(current_path_abs)
print('{} appended to sys!'.format(current_path_abs))

from utils_calc import normalize_data, normalize_data_old, perform_inference_volumetric_image, get_mcc
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
use_state_dict = False

# Load net
if use_state_dict:
    path_net = os.path.join(current_path_abs, 'logs/vessels/model_epoch_0400.pht')
    net = VXNet(dropout=True,context=2,num_outs=2)
    net.load_state_dict(torch.load(path_net))
else:
    # LAB
    # path_net = os.path.join(current_path_abs, 'logs/vessels/model_25D__2020-02-02__18_15_53.pht')
    # TESLA
    path_net = os.path.join(current_path_abs, 'logs/vessels/model_25D__2020-02-02__18_18_06.pht')
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

calculate_cnn_out = True
print('Calculate CNN Output = {}'.format(calculate_cnn_out))

if calculate_cnn_out:
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

accs       = np.zeros(len(val_list))
senss      = np.zeros(len(val_list))
specs      = np.zeros(len(val_list))

mccs       = np.zeros(len(val_list))

do_apply_liver_mask = False
print('Apply Liver Mask = {}'.format(do_apply_liver_mask))

tps, tns, fps, fns = 0,0,0,0

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

    if do_apply_liver_mask:
        # Liver GT
        gt_liver_path = os.path.join(folder_dataset, folder_patient_valid, 'mask', 'liver.nii')
        gt_liver_mask = nib.load(gt_liver_path)
        gt_liver_mask = gt_liver_mask.get_data()
        sitk_input = sitk.GetImageFromArray(gt_liver_mask)
        vector_radius = (25, 25, 25) 
        kernel = sitk.sitkBall
        gt_liver_mask_filled = sitk.BinaryMorphologicalClosing(sitk_input, vector_radius, kernel)
        print('Liver non-zero elements before filling: {}'.format(gt_liver_mask.sum()))
        gt_liver_mask = sitk.GetArrayFromImage(gt_liver_mask_filled)
        print('Liver non-zero elements after  filling: {}'.format(gt_liver_mask.sum()))

        # Filtering
        print("Output non-zero elements before: {}".format(output.sum()))
        print("GT     non-zero elements before: {}".format(gt_vessels_mask.sum()))
        output[gt_liver_mask==0] = 0
        gt_vessels_mask[gt_liver_mask==0] = 0
        print("Output non-zero elements after : {}".format(output.sum()))
        print("GT     non-zero elements after : {}".format(gt_vessels_mask.sum()))

    iou = mmb.jc(output, gt_vessels_mask)
    dice = mmb.dc(output, gt_vessels_mask)
    prec = mmb.precision(output, gt_vessels_mask)
    recall = mmb.recall(output, gt_vessels_mask)
    rvd = mmb.ravd(output, gt_vessels_mask)
    assd = mmb.assd(output, gt_vessels_mask, voxelspacing=voxel_spacing)
    hd = mmb.hd(output, gt_vessels_mask, voxelspacing=voxel_spacing)

    print('Patient   = ', folder_patient_valid)
    print('\nVolumetric Overlap Metrics')
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

    # CONFUSION MATRIX
    tn, fp, fn, tp = confusion_matrix(y_true=gt_vessels_mask.flatten(), y_pred=output.flatten()).ravel()
    acc  = (tp+tn) / (tp+tn+fp+fn)
    sens = tp / (tp+fn) # recall
    spec = tn / (tn+fp)
    # mcc = get_mcc(tp=tp, tn=tn, fp=fp, fn=fn)
    mcc = matthews_corrcoef(y_true=gt_vessels_mask.flatten(), y_pred=output.flatten())

    tps += tp
    fps += fp
    fns += fn
    tns += tn

    accs [idx] = acc
    senss[idx] = sens
    specs[idx] = spec
    mccs [idx] = int(mcc)

    print('\nConfusion Matrix Metrics')
    print('Accuracy  = {}'.format(acc))
    print('Sens (Re) = {}'.format(sens))
    print('Specif    = {}'.format(spec))
    print('MCC       = {}'.format(mcc))

avg_iou  = np.mean(ious)
avg_prec = np.mean(precisions)
avg_reca = np.mean(recalls)
avg_dice = np.mean(dices)
avg_rvd  = np.mean(rvds)
avg_assd = np.mean(assds)
avg_hd   = np.mean(hds)

avg_acc  = np.mean(accs)
avg_sens = np.mean(senss)
avg_spec = np.mean(specs)
avg_mcc  = np.mean(mccs)

print('\nVolumetric Overlap Metrics')
print('Average IoU       = {}'.format(avg_iou))
print('Average Precision = {}'.format(avg_prec))
print('Average Recall    = {}'.format(avg_reca))
print('Average Dice      = {}'.format(avg_dice))
print('Average RVD       = {}'.format(avg_rvd))
print('Average ASSD      = {}'.format(avg_assd))
print('Average MSSD      = {}'.format(avg_hd))

print('\nConfusion Matrix Metrics')
print('Average Accuracy  = {}'.format(avg_acc))
print('Average Sens (Re) = {}'.format(avg_sens))
print('Average Specif    = {}'.format(avg_spec))
print('Average MCC       = {}'.format(avg_mcc))

import json

data = {
    'IoU'       : avg_iou,
    'Precision' : avg_prec,
    'Recall'    : avg_reca,
    'Dice'      : avg_dice,
    'RVD'       : avg_rvd,
    'ASSD'      : avg_assd,
    'MSSD'      : avg_hd,

    'Acc'       : avg_acc,
    'Sens'      : avg_sens,
    'Spec'      : avg_spec,

    'Mcc'       : avg_mcc,

    'TP'        : tps,
    'FP'        : fps,
    'FN'        : fns,
    'TN'        : tns
}

json_path = os.path.join(current_path_abs, 'datasets/ircadb_metrics.json')
print('JSON Path = {}'.format(json_path))

with open(json_path, 'w') as f:
    json.dump(data, f)