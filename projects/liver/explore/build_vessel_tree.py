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

from util.geometric import get_xmin_xmax, getCentroid
from util.tree import Tree
from utils import montage, imshow_components

######################
### LOADING IMAGES ###
######################
dataset_path = 'datasets/ircadb'
patient = 'patient-{:02d}'.format(1)
pred_mask_path = os.path.join(dataset_path, patient, 'image', 'pred.nii')
image_mask_path = os.path.join(dataset_path, patient, 'image', 'image.nii')

image_sitk = sitk.ReadImage(image_mask_path)
image_np = sitk.GetArrayFromImage(image_sitk)

mask_sitk = sitk.ReadImage(pred_mask_path)
mask_np = sitk.GetArrayFromImage(mask_sitk)

zmin, zmax = get_xmin_xmax(mask_np)

#######################
### PLOTTING SLICES ###
#######################
# Show first slice where vessels appear
img_mask_start = montage([image_np[zmin,:,:], mask_np[zmin,:,:]], dim=(1,2), consistency=True, normalization=True)

plt.imshow(img_mask_start, cmap='gray')
plt.show()

# Show last slice where vessels appear
img_mask_end = montage([image_np[zmax,:,:], mask_np[zmax,:,:]], dim=(1,2), consistency=True, normalization=True)

plt.imshow(img_mask_end, cmap='gray')
plt.show()

############################
### BUILDING TREE (ROOT) ###
############################
# Starting from last slice (heart)
mask_slice_np = mask_np[zmax,:,:]
label_m1 = mask_slice_np
label_count_m1 = np.bincount(label_m1.ravel())
label_count_m1[0] = 0

area = mask_slice_np.sum()
centroid = getCentroid(mask_slice_np)
mask_slice_np_rgb = cv2.cvtColor(mask_slice_np, cv2.COLOR_GRAY2RGB)
mask_slice_np_rgb *= 255
mask_slice_np_circle = cv2.circle(mask_slice_np_rgb.copy(), centroid, 3, (0, 255, 0), -1)

plt.imshow(mask_slice_np_circle)
plt.show()

#Build Tree Root
information_struct = {
    'centroid' : centroid,
    'area' : area,
}
tree_root = Tree(name='root', information_struct=information_struct)

########################################
### BUILDING TREE (LOOP ON THE SCAN) ###
########################################

uniques = list()

for idx, slice in enumerate(mask_np):
    print('Index {} on {}'.format(idx, mask_np.shape[0]-1))
    ret, labels = cv2.connectedComponents(slice)
    uniques.append(np.unique(labels))
    print('Uniques = {}'.format(uniques[-1]))

for idx, slice in enumerate(mask_np[-15:-10,:,:]):
    ret, labels = cv2.connectedComponents(slice)
    plt.figure()
    imshow_components(labels)


#################################
### BUILDING TREE (END SLICE) ###
#################################

last_slice = mask_np[zmin,:,:].astype(np.uint8)
ret, labels = cv2.connectedComponents(last_slice)

imshow_components(labels)

cc_idxs = [idx for idx in np.unique(labels) if idx != 0]
cc_areas = list()
cc_centroids = list()

for cc_idx in cc_idxs:
    cc_idx_mask = (labels==cc_idx).astype(np.uint8)
    cc_area = cc_idx_mask.sum()
    cc_areas.append(cc_area)

    cc_centroid = getCentroid(cc_idx_mask)
    cc_centroids.append(cc_centroid)
