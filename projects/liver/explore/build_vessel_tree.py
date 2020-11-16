import os
import sys
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

#####################
### LOCAL IMPORTS ###
#####################
abs_dir = os.path.abspath('.')
sys.path.append(abs_dir)

from util.geometric import get_xmin_xmax, getCentroid, getEuclidanDistance
from util.tree import TreeNode, TreeNodesContainer, label_tree_from_branches
from utils import montage, imshow_components

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


#####################################################
### BUILDING TREE (FROM END SLICE TO START SLICE) ###
#####################################################
do_print_to_file_tree = True
show_comps = False
use_gt = True
nodes_container = TreeNodesContainer()
mask_labelled_cv2 = np.zeros(mask_gt_hv_np.shape, np.uint8)

for idx_slice in range(zmax, zmin, -1):

    last_slice = mask_gt_hv_np[idx_slice,:,:].astype(np.uint8)
    ret, labels = cv2.connectedComponents(last_slice)
    mask_labelled_cv2[idx_slice,:,:] = labels

    if show_comps:
        plt.figure()
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

        information_struct = {
            'centroid': cc_centroid,
            'area': cc_area,
        }
        new_node = TreeNode(name='s_{}_{}'.format(idx_slice, cc_idx), information_struct=information_struct)
        nodes_container.add_node(new_node)

#################################
### LINKING NODES OF THE TREE ###
#################################

#####################
### RIGHT VERSION ###
#####################
print('Printing layers contained in container')
layers = nodes_container.get_layers()
for layer in layers:
    print('Layer = {}'.format(layer))

layer_old = layers[0]

for layer in layers[1:]:
    layer_new = layer

    distance_matrix = nodes_container.build_distance_matrix_between_layers(layer_new, layer_old)
    print('Distance Matrix\n{}'.format(distance_matrix))

    amins = list()
    for idx, row in enumerate(distance_matrix):
        amin = np.argmin(row)
        print('ArgMin = {}'.format(amin))
        amins.append(amin)
        parent_node_name = 's_{}_{}'.format(layer_old, amin+1)
        child_node_name = 's_{}_{}'.format(layer_new, idx+1)
        print('Link {} ---> {}'.format(parent_node_name, child_node_name))
        nodes_container.link_two_nodes(parent_node_name, child_node_name)
        parent_node = nodes_container.get_node_from_name(parent_node_name)

    layer_old = layer_new

print('\n\n\nPrint Tree!\n')
root_node = nodes_container.nodes[0]
print(root_node)

root_node.set_recursively_depth()
threshold_new_branch = 50
bb = root_node.recursively_find_branches_tree(threshold_new_branch=threshold_new_branch, do_init=True)
# label_tree_from_branches(bb)
print('Branches = {}'.format(bb))
print(root_node.get_string_label())
print(root_node.get_string_label(limit=20))

if do_print_to_file_tree:
    file_tree = open("tree.txt", 'w')
    file_tree.write(str(root_node))
    file_tree.close()

    file_tree_label = open("tree_label.txt", 'w')
    file_tree_label.write(root_node.get_string_label(limit=60))
    file_tree_label.close()

####################################
### MAP LABELED TREE TO 3D IMAGE
####################################

mask_labelled = np.zeros(mask_gt_hv_np.shape, np.uint8)

bb_labels = [bb_.name for bb_ in bb]
def get_index_from_label(idx, bb_labels):
    return bb_labels.index(idx)+1

for idx_slice in range(zmax, zmin, -1):

    # Use previous labeled connected components
    labels = mask_labelled_cv2[idx_slice,:,:]

    # Label connected components using cv2
    # last_slice = mask_gt_hv_np[idx_slice,:,:].astype(np.uint8)
    # ret, labels = cv2.connectedComponents(last_slice)

    cc_idxs = [idx for idx in np.unique(labels) if idx != 0]

    # Get corresponding slide
    for cc_idx in cc_idxs:
        node_name = 's_{}_{}'.format(idx_slice, cc_idx)
        branch_label = nodes_container.get_node_from_name(node_name).label_sitk
        branch_label = 's_128_1' if branch_label == 'main_branch' else branch_label
        labels[labels==cc_idx] = get_index_from_label(branch_label, bb_labels)

    mask_labelled[idx_slice,:,:] = labels

print('Unique values of Final Mask = {}'.format(np.unique(mask_labelled)))

mask_labelled_sitk = sitk.GetImageFromArray(mask_labelled)
mask_labelled_sitk.SetSpacing(mask_gt_hv_sitk.GetSpacing())

path_labelled_mask = os.path.join(gt_mask_hv_path[:-7], 'hv_lab.nii')
sitk.WriteImage(mask_labelled_sitk, path_labelled_mask)



####################################
### GET 3D CENTROIDS AND CLUSTER ###
####################################
centroids_all = list()
for node in nodes_container.nodes:
    centroid_2d = node.information_struct['centroid']
    _, layer, _ = node.name.split('_')
    centroid_3d = (centroid_2d[0], centroid_2d[1], layer)
    centroids_all.append(centroid_3d)

X = np.zeros((len(centroids_all),3))

for idx, centroid_3d in enumerate(centroids_all):
    x,y,z = centroid_3d
    X[idx,:] = [x,y,z]


##########
### V2 ###
##########
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='single')
cluster.fit_predict(X)
print(cluster.labels_)
for idx, node in enumerate(nodes_container.nodes):
    label_from_clustering = cluster.labels_[idx]+1
    node.label_sitk = label_from_clustering

####
labels_ = set()
for node in nodes_container.nodes:
    label_ = node.label_sitk
    labels_.add(label_)

################
### LABELING ###
################

mask_labelled = np.zeros(mask_gt_hv_np.shape, np.uint8)

for idx_slice in range(zmax, zmin, -1):

    # Use previous labeled connected components
    labels = mask_labelled_cv2[idx_slice,:,:]
    cc_idxs = [idx for idx in np.unique(labels) if idx != 0]

    # Get corresponding slide
    for cc_idx in cc_idxs:
        node_name = 's_{}_{}'.format(idx_slice, cc_idx)
        label_val = nodes_container.get_node_from_name(node_name).label_sitk
        labels[labels==cc_idx] = label_val

    mask_labelled[idx_slice,:,:] = labels

print('Unique values of Final Mask = {}'.format(np.unique(mask_labelled)))

mask_labelled_sitk = sitk.GetImageFromArray(mask_labelled)
mask_labelled_sitk.SetSpacing(mask_gt_hv_sitk.GetSpacing())

path_labelled_mask = os.path.join(gt_mask_hv_path[:-7], 'hv_lab.nii')
sitk.WriteImage(mask_labelled_sitk, path_labelled_mask)

##########
### V1 ###
##########
linked = linkage(X, 'single')
label_list = np.array([idx for idx in range(1,5)])

plt.figure()
dendrogram(linked, orientation='top',
           # labels=label_list,
           distance_sort='descending',
           show_leaf_counts=True)
plt.show()

##################
### TODO: Delete
##################
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
tree_root = TreeNode(name='root', information_struct=information_struct)



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


###################
### TODO: Delete
###################
print('Start Printing all nodes!')
for node in nodes_container.nodes:
    print(node)
print('End   Printing all nodes!')

print('Printing layers contained in container')
layers = nodes_container.get_layers()
for layer in layers:
    print('Layer = {}'.format(layer))

layer_to_cmp = '110'
nodes_of_layer = nodes_container.get_nodes_of_layer(layer_to_cmp)

print('Printing all nodes in layer_to_cmp={}'.format(layer_to_cmp))
for node in nodes_of_layer:
    print(node)


####################
### TODO: Delete
####################
start_layer = layers[0]
start_nodes = nodes_container.nodes[0]
for layer in layers[1:2]:
    nodes_of_layer = nodes_container.get_nodes_of_layer(layer)
    centroids_of_layer = nodes_container.get_key_for_nodes_of_layer(layer, 'centroid')
    print('Layer {}'.format(layer))
    print('Nodes\n{}'.format(nodes_of_layer))
    print('Centroids\n{}'.format(centroids_of_layer))

#TODO: delete this section!
###################
### OLD VERSION ###
###################
centroidss = list()

for layer in layers:
    centroids = list()
    for node in nodes_container.nodes:
        if node.name[6] == layer:
            centroids.append(node.information_struct['centroid'])
    centroidss.append(centroids)

centroids_last_slice = centroidss[-1]
centroids_second_last_slice = centroidss[-2]

distance_matrix = np.zeros((len(centroids_last_slice), len(centroids_second_last_slice)))

for idx_ls, centroid_ls in enumerate(centroids_last_slice):
    for idx_sls, centroid_sls in enumerate(centroids_second_last_slice):
        distance_matrix[idx_ls, idx_sls] = getEuclidanDistance(centroids_last_slice[idx_ls],
                                                               centroids_second_last_slice[idx_sls])
amins = list()
for idx, row in enumerate(distance_matrix):
    amin = np.argmin(row)
    print('ArgMin = {}'.format(amin))
    amins.append(amin)
    print('Link Second Last Slice ({}) with Last Slice ({})'.format(amin+1,idx+1))