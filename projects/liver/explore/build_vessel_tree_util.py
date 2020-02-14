import os

import SimpleITK as sitk
import numpy as np

from util.geometric import get_xmin_xmax
from utils_calc import get_iou


def get_largest_cc(labels_old):
    cc_idx_masks = list()
    cc_areas = list()
    cc_idxs = [idx for idx in np.unique(labels_old) if idx != 0]

    for cc_idx in cc_idxs:
        cc_idx_mask = (labels_old == cc_idx).astype(np.uint8)
        cc_idx_masks.append(cc_idx_mask)

        cc_area = cc_idx_mask.sum()
        cc_areas.append(cc_area)

    if len(cc_areas) == 0:
        print('cc_idxs = {}'.format(cc_idxs))
        return None
    else:
        cc_idx_max_area = np.argmax(cc_areas)
        cc_mask_max_area_old = cc_idx_masks[cc_idx_max_area]
        return cc_mask_max_area_old


def check_if_is_main_branch(mask_cv2, threshold_area=5000, threshold_slices=10):
    zmin, zmax = get_xmin_xmax(mask_cv2)
    num_slices_branch = zmax-zmin
    area_voxels = np.sum(mask_cv2)
    print('Area [voxels]     = {}'.format(area_voxels))
    print('Num Slices Branch = {}'.format(num_slices_branch))
    return  area_voxels > threshold_area or num_slices_branch > threshold_slices


def get_largest_overlap_label(labels_new, labels_old_largest_cc, th_iou=0.1):
    cc_idxs = [idx for idx in np.unique(labels_new) if idx != 0]
    cc_areas = list()
    cc_idx_masks = list()
    cc_ious = list()

    for cc_idx in cc_idxs:
        cc_idx_mask = (labels_new == cc_idx).astype(np.uint8)
        cc_idx_masks.append(cc_idx_mask)

        cc_area = cc_idx_mask.sum()
        cc_areas.append(cc_area)

        cc_iou = get_iou(labels_old_largest_cc, cc_idx_mask, smooth=0)
        cc_ious.append(cc_iou)

    cc_idx_max_iou = np.argmax(cc_ious)
    cc_mask_max_iou = cc_idx_masks[cc_idx_max_iou]

    cc_max_iou = cc_ious[cc_idx_max_iou]

    labels_old_largest_cc = cc_mask_max_iou
    is_not_none = False

    if sum(cc_ious) > 0:
        is_not_none = True
    if cc_max_iou < th_iou:
        is_not_none = False
        print('Max IoU = {}'.format(cc_max_iou))

    return labels_old_largest_cc, is_not_none


def load_from_ircadb(dataset_path, idx_ircad):
    patient = 'patient-{:02d}'.format(idx_ircad)
    gt_mask_liver_path = os.path.join(dataset_path, patient, 'mask', 'liver.nii')
    gt_mask_hv_path = os.path.join(dataset_path, patient, 'mask', 'hv.nii')
    gt_mask_pv_path = os.path.join(dataset_path, patient, 'mask', 'pv.nii')
    pred_mask_path = os.path.join(dataset_path, patient, 'image', 'pred.nii')
    image_mask_path = os.path.join(dataset_path, patient, 'image', 'image.nii')

    image_sitk = sitk.ReadImage(image_mask_path)
    image_np = sitk.GetArrayFromImage(image_sitk)

    mask_pred_sitk = sitk.ReadImage(pred_mask_path)
    mask_pred_np = sitk.GetArrayFromImage(mask_pred_sitk)

    mask_gt_hv_sitk = sitk.ReadImage(gt_mask_hv_path)
    mask_gt_hv_np = sitk.GetArrayFromImage(mask_gt_hv_sitk)

    mask_gt_pv_sitk = sitk.ReadImage(gt_mask_pv_path)
    mask_gt_pv_np = sitk.GetArrayFromImage(mask_gt_pv_sitk)

    mask_gt_np = np.logical_or(mask_gt_hv_np, mask_gt_pv_np)

    mask_gt_liver_sitk = sitk.ReadImage(gt_mask_liver_path)
    mask_gt_liver_np = sitk.GetArrayFromImage(mask_gt_liver_sitk)
    return mask_gt_hv_sitk, gt_mask_hv_path, image_np, mask_pred_np, mask_gt_np, mask_gt_liver_np
