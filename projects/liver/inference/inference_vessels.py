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
net_hv = torch.load(path_net_hv)
net_pv = torch.load(path_net_pv)

# Start iteration over val set
for folder_patient_valid in folders_patients_valid:

    path_test_image = os.path.join(folder_dataset, folder_patient_valid, 'image', 'image.nii')
    path_test_pred  = os.path.join(folder_dataset, folder_patient_valid, 'image', 'pred.nii')
    path_test_pred_hv = os.path.join(folder_dataset, folder_patient_valid, 'image', 'pred_hv.nii')
    path_test_pred_pv = os.path.join(folder_dataset, folder_patient_valid, 'image', 'pred_pv.nii')

    # load file
    data = nib.load(path_test_image)
    # save affine
    input_aff = data.affine
    # convert to numpy
    data = data.get_data()

    # normalize data
    data = normalize_data(data, dmin=window_hu[0], dmax=window_hu[1])

    # transpose so the z-axis (slices) are the first dimension
    data = np.transpose(data, (2, 0, 1))
    # CNN
    output = perform_inference_volumetric_image(net, data, context=2, do_round=True)
    output = np.transpose(output, (1, 2, 0)).astype(np.uint8)

    # HV / PV
    output_hv = perform_inference_volumetric_image(net_hv, data, context=2, do_round=True)
    output_pv = perform_inference_volumetric_image(net_pv, data, context=2, do_round=True)
    output_hv = np.transpose(output_hv, (1, 2, 0)).astype(np.uint8)
    output_pv = np.transpose(output_pv, (1, 2, 0)).astype(np.uint8)

    n_nonzero = np.count_nonzero(output)
    print("Non-zero elements = ", n_nonzero)

    output_nib_hv = nib.Nifti1Image(output_hv, affine=input_aff)
    output_nib_pv = nib.Nifti1Image(output_pv, affine=input_aff)
    nib.save(output_nib_hv, path_test_pred_hv)
    nib.save(output_nib_pv, path_test_pred_pv)

    output_nib_pre = nib.Nifti1Image(output, affine=input_aff)
    nib.save(output_nib_pre, path_test_pred)

    gt_pv_path = os.path.join(folder_dataset, folder_patient_valid, 'mask', 'pv.nii')
    gt_hv_path = os.path.join(folder_dataset, folder_patient_valid, 'mask', 'hv.nii')

    gt_pv_mask = nib.load(gt_pv_path)
    gt_pv_mask = gt_pv_mask.get_data()
    gt_hv_mask = nib.load(gt_hv_path)
    gt_hv_mask = gt_hv_mask.get_data()

    gt_vessels_mask = gt_pv_mask + gt_hv_mask

    iou = mmb.jc(output, gt_vessels_mask)
    dice = mmb.dc(output, gt_vessels_mask)
    prec = mmb.precision(output, gt_vessels_mask)
    recall = mmb.recall(output, gt_vessels_mask)
    rvd = mmb.ravd(output, gt_vessels_mask)

    print('Patient   = ', folder_patient_valid)
    print('IoU       = ', iou)
    print('Dice      = ', dice)
    print('Precision = ', prec)
    print('Recall    = ', recall)
    print('RVD       = ', rvd)

    iou_hv = mmb.jc(output_hv, gt_hv_mask)
    dice_hv = mmb.dc(output_hv, gt_hv_mask)
    prec_hv = mmb.precision(output_hv, gt_hv_mask)
    recall_hv = mmb.recall(output_hv, gt_hv_mask)
    rvd_hv = mmb.ravd(output_hv, gt_hv_mask)

    print('Hepatic Vein')
    print('IoU       = ', iou_hv)
    print('Dice      = ', dice_hv)
    print('Precision = ', prec_hv)
    print('Recall    = ', recall_hv)
    print('RVD       = ', rvd_hv)

    iou_pv = mmb.jc(output_hv, gt_hv_mask)
    dice_pv = mmb.dc(output_hv, gt_hv_mask)
    prec_pv = mmb.precision(output_hv, gt_hv_mask)
    recall_pv = mmb.recall(output_hv, gt_hv_mask)
    rvd_pv = mmb.ravd(output_hv, gt_hv_mask)

    print('Portal Vein')
    print('IoU       = ', iou_pv)
    print('Dice      = ', dice_pv)
    print('Precision = ', prec_pv)
    print('Recall    = ', recall_pv)
    print('RVD       = ', rvd_pv)

# TODO: post processing
# Connected components labeling
# Get connected components
ccif = sitk.ConnectedComponentImageFilter()
sitk_output = sitk.GetImageFromArray(output)
conn_comps = ccif.Execute(sitk_output)
conn_comps_np = sitk.GetArrayFromImage(conn_comps)

unique_values = np.unique(conn_comps_np)
n_uniques = len(unique_values)
counter_uniques = np.zeros(n_uniques)
for i in range(1, max(unique_values) + 1):
    counter_uniques[i] = (conn_comps_np == i).sum()
biggest_region_value = np.argmax(counter_uniques)

# Morphological closing
vector_radius = (10,10,10)
kernel = sitk.sitkBall
output_uint8 = output.astype(np.uint8)
output_sitk = sitk.GetImageFromArray(output_uint8)
output_closed_sitk = sitk.BinaryMorphologicalClosing(output_sitk, vector_radius, kernel)
output_closed_np = sitk.GetArrayFromImage(output_closed_sitk)

conn_comps_closed = ccif.Execute(output_closed_sitk)
conn_comps_closed_np = sitk.GetArrayFromImage(conn_comps_closed)

# Morphological opening
vector_radius = (10,10,10)
kernel = sitk.sitkBall
# morph_open_out = sitk.BinaryMorphologicalOpening(conn_comps_closed, vector_radius, kernel)
morph_open_out = sitk.BinaryMorphologicalOpening(output_sitk, vector_radius, kernel)

conn_comps_opened = ccif.Execute(morph_open_out)
conn_comps_opened_np = sitk.GetArrayFromImage(conn_comps_opened)

#
unique_values = np.unique(conn_comps_opened_np)
n_uniques = len(unique_values)
counter_uniques = np.zeros(n_uniques)
for i in range(1, max(unique_values) + 1):
    counter_uniques[i] = (conn_comps_closed_np == i).sum()
biggest_1_region_value = np.argmax(counter_uniques)

counter_uniques_without_max = np.delete(counter_uniques, biggest_region_value)
biggest_2_region_value = np.argmax(counter_uniques_without_max)