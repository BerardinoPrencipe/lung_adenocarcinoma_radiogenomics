import os
import sys
import numpy as np
import SimpleITK as sitk
import cv2

#####################
### LOCAL IMPORTS ###
#####################
abs_dir = os.path.abspath('.')
sys.path.append(abs_dir)

from util.geometric import get_xmin_xmax, getCentroid
from projects.liver.explore.build_vessel_tree_util import get_largest_cc, check_if_is_main_branch, \
    get_largest_overlap_label, load_from_ircadb
from utils import print_matrix

######################
### LOADING IMAGES ###
######################
dataset_path = os.path.join(abs_dir, 'datasets', 'ircadb')
idx_ircad = 2
mask_gt_hv_sitk, gt_mask_hv_path, image_np, \
mask_pred_np, mask_gt_np, mask_gt_liver_np = load_from_ircadb(dataset_path, idx_ircad)
path_labelled_mask = os.path.join(gt_mask_hv_path[:-7], 'hv_main_branches.nii')

path_dataset_scardapane = 'E:/Datasets/LiverScardapane'
patient_name = '03_Leone'
path_nifti_pred_scardapane = os.path.join(path_dataset_scardapane, patient_name+'.nii')
path_nifti_gt_scardapane = os.path.join(path_dataset_scardapane, patient_name, 'mask.nii.gz')
# mask_gt_sitk = sitk.ReadImage(path_nifti_pred_scardapane)
mask_gt_sitk = sitk.ReadImage(path_nifti_gt_scardapane)
mask_gt_np = sitk.GetArrayFromImage(mask_gt_sitk)
path_labelled_mask = os.path.join(path_dataset_scardapane, patient_name+"_annotated.nii")

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

    zmin, zmax = get_xmin_xmax(mask_gt_updated)
    mask_assigned_curr_cv2 = np.zeros(mask_gt_updated.shape, np.uint8)

    first_slice = mask_gt_updated[zmax,:,:].astype(np.uint8)
    ret_old,labels_old = cv2.connectedComponents(first_slice)

    labels_old_largest_cc = get_largest_cc(labels_old)
    if labels_old_largest_cc is None:
        print('Index Branch = {}'.format(idx_branch))
        print('Largest CC is None!')
        print('Z range = [{} {}]'.format(zmin, zmax))
        break

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


    if check_if_is_main_branch(mask_assigned_curr_cv2):
        idx_branch += 1

        mask_main_branches = mask_main_branches + mask_assigned_curr_cv2 * idx_branch

        if idx_branch == num_max_branches:
            print('Found all main branches! Terminating.')
    else:
        patience += 1
        print('Criteria not matching! Patience = {}'.format(patience))

    if mask_gt_updated.sum() == 0:
        print('No more voxels in mask')
        break


#########################
### COUNT CC ELEMENTS ###
#########################

unique_values = np.unique(mask_main_branches)
unique_values_no_zero = [x for x in unique_values if x != 0]
counter_masks = list()
for val in unique_values_no_zero:
    cnt_mask_val = (mask_main_branches == val).sum()
    print('Val = {} --> Voxels = {}'.format(val, cnt_mask_val))
    counter_masks.append(cnt_mask_val)

########################
### MERGING BRANCHES ###
########################
do_merge = False
if do_merge:
    from medpy.metric.binary import assd
    unique_values = np.unique(mask_main_branches)
    unique_values_no_zero = [x for x in unique_values if x != 0]
    mask_vals = list()
    for val in unique_values_no_zero:
        mask_val = (mask_main_branches == val)
        mask_vals.append(mask_val)

    MAX_VALUE = 10000
    distances_matrix = np.ones((len(mask_vals), len(mask_vals))) * MAX_VALUE
    for i in range(len(mask_vals)):
        for j in range(i+1,len(mask_vals)):
            print('Calculating d({})({})'.format(i, j))
            distances_matrix[i,j] = assd(mask_vals[i], mask_vals[j])

    print_matrix(distances_matrix)

    amin_linear = np.argmin(distances_matrix)
    amin_matrix = np.unravel_index(amin_linear, distances_matrix.shape)
    print('It is suggested to merge branch {} with branch {}'.format(amin_matrix[0], amin_matrix[1]))

    merge_idx_1, merge_idx_2 = amin_matrix
    merge_idx_1, merge_idx_2 = merge_idx_1+1,merge_idx_2+1

    mask_main_branches_after_merge = mask_main_branches.copy()
    mask_main_branches_after_merge[mask_main_branches_after_merge==merge_idx_1] = merge_idx_2

    print('Branches merged!')

##############################
### SAVING RESULT TO IMAGE ###
##############################
if do_merge:
    mask_labelled_sitk = sitk.GetImageFromArray(mask_main_branches_after_merge)
else:
    mask_labelled_sitk = sitk.GetImageFromArray(mask_main_branches)
mask_labelled_sitk.SetSpacing(mask_gt_hv_sitk.GetSpacing())

sitk.WriteImage(mask_labelled_sitk, path_labelled_mask)

###################################
### GET COORDINATES OF BRANCHES ###
###################################
unique_values = np.unique(mask_main_branches)
unique_values_no_zero = [x for x in unique_values if x != 0]
slices_fitting = list()
centroidss = list()
for val in unique_values_no_zero:
    centroids  = list()
    slice_fits = list()
    mask_val = (mask_main_branches == val).astype(np.uint8)
    z_start_mask, z_end_mask = get_xmin_xmax(mask_val)
    z_end_mask -= 3
    slice_fit = mask_val[z_end_mask]
    centroid = getCentroid(slice_fit)
    slice_fits.append(slice_fit)
    centroids.append(centroid)
    z_end_mask -= 3
    slice_fit = mask_val[z_end_mask]
    centroid = getCentroid(slice_fit)
    slice_fits.append(slice_fit)
    centroids.append(centroid)

    centroidss.append(centroids)
    slices_fitting.append(slice_fits)

''' 
plane_sitk = sitk.GetImageFromArray(points)
plane_sitk.SetSpacing(mask_gt_hv_sitk.GetSpacing())

path_plane_mask = os.path.join(gt_mask_hv_path[:-7], 'plane.nii')
sitk.WriteImage(plane_sitk, path_plane_mask)
'''

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





'''
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
        # X[i, :] = coord[i][1:3]
        # Y[i] = coord[i][0]
        X[i, :] = coord[i][0:2]
        Y[i] = coord[i][2]
    Xs.append(X)
    Ys.append(Y)

from sklearn.linear_model import LinearRegression
models = list()
for X,Y in zip(Xs, Ys):
    model = LinearRegression().fit(X,Y)
    r_sq = model.score(X,Y)
    print('Coefficient of Determination: {}'.format(r_sq))
    models.append(model)

points = np.zeros(mask_gt_hv_np.shape)
model = models[1]
dim_z = points.shape[0]
dim_xy = 512
# for x in range(dim_xy):
for x in range(dim_z):
    for y in range(dim_xy):
        # z = int(np.ceil(model.predict([[x,y]])))
        # z = np.clip(z, 0, points.shape[0]-1)
        z = int(np.ceil(model.predict([[x,y]])))
        z = np.clip(z, 0, points.shape[0]-1)
        points[x,y,z] = 1

plane_sitk = sitk.GetImageFromArray(points)
plane_sitk.SetSpacing(mask_gt_hv_sitk.GetSpacing())

path_plane_mask = os.path.join(gt_mask_hv_path[:-7], 'plane.nii')
sitk.WriteImage(plane_sitk, path_plane_mask)
'''