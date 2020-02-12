import os
import sys
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import matplotlib.pyplot as plt
import cv2

#####################
### LOCAL IMPORTS ###
#####################
abs_dir = os.path.abspath('.')
sys.path.append(abs_dir)

from util.geometric import get_xmin_xmax, getCentroid, getEuclidanDistance
from utils import montage, imshow_components
from utils_calc import get_iou

######################
### LOADING IMAGES ###
######################
dataset_path = 'datasets/ircadb'
patient = 'patient-{:02d}'.format(1)
gt_mask_hv_path = os.path.join(dataset_path, patient, 'mask', 'hv.nii')
gt_mask_pv_path = os.path.join(dataset_path, patient, 'mask', 'pv.nii')
pred_mask_path = os.path.join(dataset_path, patient, 'image', 'pred.nii')
image_mask_path = os.path.join(dataset_path, patient, 'image', 'image.nii')

image_sitk = sitk.ReadImage(image_mask_path)
image_np = sitk.GetArrayFromImage(image_sitk)

mask_sitk = sitk.ReadImage(pred_mask_path)
mask_np = sitk.GetArrayFromImage(mask_sitk)

mask_gt_hv_sitk = sitk.ReadImage(gt_mask_hv_path)
mask_gt_hv_np = sitk.GetArrayFromImage(mask_gt_hv_sitk)

mask_gt_pv_sitk = sitk.ReadImage(gt_mask_pv_path)
mask_gt_pv_np = sitk.GetArrayFromImage(mask_gt_pv_sitk)

mask_gt_np = np.logical_or(mask_gt_hv_np, mask_gt_pv_np)

zmin, zmax = get_xmin_xmax(mask_np)


def get_largest_cc(labels_old):
    cc_idx_masks = list()
    cc_areas = list()
    cc_idxs = [idx for idx in np.unique(labels_old) if idx != 0]

    for cc_idx in cc_idxs:
        cc_idx_mask = (labels_old == cc_idx).astype(np.uint8)
        cc_idx_masks.append(cc_idx_mask)

        cc_area = cc_idx_mask.sum()
        cc_areas.append(cc_area)

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

#####################################################
### BUILDING TREE (FROM END SLICE TO START SLICE) ###
#####################################################

# FLAGS
show_comps = False

################
## OUTER LOOP ##
################
num_max_branches = 5
max_patience = 50
idx_branch = 0
patience = 0

# mask_gt_updated = mask_gt_hv_np.copy().astype(np.uint8)
mask_gt_updated = mask_gt_np.copy().astype(np.uint8)
mask_main_branches = np.zeros(mask_gt_updated.shape, np.uint8)

while(idx_branch < num_max_branches and patience < max_patience):

    mask_assigned_curr_cv2 = np.zeros(mask_gt_updated.shape, np.uint8)

    first_slice = mask_gt_updated[zmax,:,:].astype(np.uint8)
    ret_old,labels_old = cv2.connectedComponents(first_slice)

    labels_old_largest_cc = get_largest_cc(labels_old)

    ################
    ## INNER LOOP ##
    ################
    for idx_slice in range(zmax, zmin, -1):
        last_slice = mask_gt_updated[idx_slice,:,:].astype(np.uint8)
        if last_slice.sum() == 0:
            print('Last slice has no voxels. Index Slice {}'.format(idx_slice))
            break
        ret_new, labels_new = cv2.connectedComponents(last_slice)
        labels_old_largest_cc, is_not_none = get_largest_overlap_label(labels_new, labels_old_largest_cc)
        if is_not_none:
            mask_assigned_curr_cv2[idx_slice, :, :] = labels_old_largest_cc
        else:
            print('No CC. Index Slice {}'.format(idx_slice))
            break

    mask_gt_updated = np.logical_xor(mask_assigned_curr_cv2, mask_gt_updated)
    zmin, zmax = get_xmin_xmax(mask_gt_updated)

    if check_if_is_main_branch(mask_assigned_curr_cv2):
        idx_branch += 1

        mask_main_branches = mask_main_branches + mask_assigned_curr_cv2 * idx_branch

        if idx_branch == 4:
            print('Found all main branches! Terminating.')
    else:
        patience += 1
        print('Criteria not matching! Patience = {}'.format(patience))

    if mask_gt_updated.sum() == 0:
        print('No more voxels in mask')
        break


mask_labelled_sitk = sitk.GetImageFromArray(mask_main_branches)
mask_labelled_sitk.SetSpacing(mask_gt_hv_sitk.GetSpacing())

path_labelled_mask = os.path.join(gt_mask_hv_path[:-7], 'hv_main_branches.nii')
sitk.WriteImage(mask_labelled_sitk, path_labelled_mask)

###################################
### GET COORDINATES OF BRANCHES ###
###################################
unique_values = np.unique(mask_main_branches)
unique_values_no_zero = [x for x in unique_values if x != 0]
coords = list()
for val in unique_values_no_zero:
    mask_val = (mask_main_branches == val)
    coord = np.argwhere(mask_val)
    coords.append(coord)

Xs = list()
Ys = list()
for coord in coords:
    X = np.zeros((len(coord),2))
    Y = np.zeros(len(coord))
    n_rows = len(coord)
    for i in range(1,n_rows):
        X[i, :] = coord[i][1:3]
        Y[i] = coord[i][0]
    Xs.append(X)
    Ys.append(Y)

from sklearn.linear_model import LinearRegression
models = list()
for X,Y in zip(Xs, Ys):
    model = LinearRegression().fit(X,Y)
    r_sq = model.score(X,Y)
    print('Coefficient of Determination: {}'.format(r_sq))
    models.append(model)

model = models[1]
dim_xy = 512
points = np.zeros(mask_gt_hv_np.shape)
for x in range(dim_xy):
    for y in range(dim_xy):
        z = int(np.ceil(model.predict([[x,y]])))
        if z >= points.shape[0]:
            z = points.shape[0]-1
        points[z,y,x] = 1

plane_sitk = sitk.GetImageFromArray(points)
plane_sitk.SetSpacing(mask_gt_hv_sitk.GetSpacing())

path_plane_mask = os.path.join(gt_mask_hv_path[:-7], 'plane.nii')
sitk.WriteImage(plane_sitk, path_plane_mask)

########################
### FIND PORTAL VEIN ###
########################
mask_gt_updated_sitk = sitk.GetImageFromArray(mask_gt_updated.astype(np.uint8))
mask_gt_updated_sitk.SetSpacing(mask_gt_hv_sitk.GetSpacing())

# Get connected components
ccif = sitk.ConnectedComponentImageFilter()
conn_comps = ccif.Execute(mask_gt_updated_sitk)
conn_comps_np = sitk.GetArrayFromImage(conn_comps)

unique_values = np.unique(conn_comps_np)
n_uniques = len(unique_values)
counter_uniques = np.zeros(n_uniques)
for i in range(1, max(unique_values) + 1):
    counter_uniques[i] = (conn_comps_np == i).sum()
biggest_region_value = np.argmax(counter_uniques)

# Get largest connected component
largest_conn_comp = np.zeros(conn_comps_np.shape, dtype=np.uint8)
largest_conn_comp[conn_comps_np == biggest_region_value] = 1

# Save PV Image
largest_conn_comp_sitk = sitk.GetImageFromArray(largest_conn_comp)
largest_conn_comp_sitk.SetSpacing(mask_gt_hv_sitk.GetSpacing())
path_labelled_mask = os.path.join(gt_mask_hv_path[:-7], 'pv_branch.nii')
sitk.WriteImage(largest_conn_comp_sitk, path_labelled_mask)