import os
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import torch
import medpy.metric.binary as mmb
import sys
import time
from sklearn.metrics import confusion_matrix, matthews_corrcoef

current_path_abs = os.path.abspath('.')
sys.path.append(current_path_abs)
print('{} appended to sys!'.format(current_path_abs))

from utils_calc import normalize_data, normalize_data_old, get_mcc
from projects.liver.util.inference import perform_inference_volumetric_image, map_thickness_to_spacing_context
from projects.liver.train.config import window_hu
from semseg.models.vnet_v2 import VXNet

do_use_spacing_context  = False
do_mask_liver           = False
use_state_dict          = False

# Load net
path_net = os.path.join(current_path_abs, 'logs/vessels_tumors/model_25D__2020-02-20__06_53_17.pht')
net = torch.load(path_net)
print('Network Path = {}'.format(path_net))

if torch.cuda.device_count() > 1:
    cuda_dev = torch.device('cuda:1')
else:
    cuda_dev = torch.device('cuda')
print('Device Count = {}, using CUDA Device = {}'.format(torch.cuda.device_count(), cuda_dev))

net = net.cuda(cuda_dev)
net.eval()

folder_dataset        = 'datasets/LiverDecathlon'
folder_input_images   = os.path.join(folder_dataset, 'nii/images' )
folder_liver_masks    = os.path.join(folder_dataset, 'nii/labels_liver' )
folder_vessels_tumors_masks = os.path.join(folder_dataset, 'nii/labels_vessels_tumors')
folder_prediction_vessels_tumors = os.path.join(folder_dataset, 'nii/pred_vessels_tumors')

if not os.path.exists(folder_prediction_vessels_tumors):
    os.makedirs(folder_prediction_vessels_tumors)

image_paths = []
for idx in range(1,50):
    image_path = 'hepaticvessel_{:03d}.nii.gz'.format(idx)
    if image_path in os.listdir(folder_vessels_tumors_masks):
        image_paths.append(image_path)
    else:
        print('Image Path {} not in Ground Truth Vessels Tumors'.format(image_path))

# Start iteration over val set
for idx, image_path in enumerate(image_paths):

    path_test_image = os.path.join(folder_input_images, image_path)
    path_test_pred_vessels_tumors = os.path.join(folder_prediction_vessels_tumors, image_path)
    path_test_liver = os.path.join(folder_liver_masks, image_path)

    # load file
    data = nib.load(path_test_image)
    thickness = data.header.get_zooms()[2]
    # save affine
    input_aff = data.affine
    # convert to numpy
    data = data.get_data()

    if do_mask_liver:
        data_liver = nib.load(path_test_liver)
        data_liver = data_liver.get_data()
        data *= data_liver

    # normalize data
    data = normalize_data(data, window_hu)

    # transpose so the z-axis (slices) are the first dimension
    data = np.transpose(data, (2, 0, 1))

    # CNN

    if do_use_spacing_context:
        spacing_context = map_thickness_to_spacing_context(thickness)
    else:
        spacing_context = 1
    print('Thickness = {} Spacing Context = {}'.format(thickness, spacing_context))
    output_vessels_tumors = perform_inference_volumetric_image(net, data, context=2,
                                                               spacing_context=spacing_context,
                                                               do_round=False, do_argmax=True, cuda_dev=cuda_dev)
    print('Shape before correction: {}'.format(output_vessels_tumors.shape))

    output_vessels_tumors = np.transpose(output_vessels_tumors, (1, 2, 0)).astype(np.uint8)

    non_zero_elements_vessels_tumors = np.count_nonzero(output_vessels_tumors)
    print("Non-Zero elements vessels tumors = {}".format(non_zero_elements_vessels_tumors))

    output_nib_vessels_tumors = nib.Nifti1Image(output_vessels_tumors, affine=input_aff)
    nib.save(output_nib_vessels_tumors, path_test_pred_vessels_tumors)

    print("Index {} on {}.\n"
          "Vessels and Tumors Image saved in {}  \n".format(idx, len(image_paths) - 1,
                                              path_test_pred_vessels_tumors))


#########################
### CALCULATE METRICS ###
#########################

from sklearn.metrics import classification_report

NUM_CLASSES = 3
NUM_IMAGES  = len(image_paths)
eps         = 1e-5
do_class_report = False

target_names = ['b', 'vessels', 'tumors']
labels = [idx for idx in range(NUM_CLASSES)]

cr_before_list = []

intersect_before = np.zeros(NUM_CLASSES)
union_before     = np.zeros(NUM_CLASSES)

accuracy_before_cumul = 0

# Start iteration over val set
for idx, image_path in enumerate(image_paths):
    print('Starting Iter {} on {}...'.format(idx, len(image_paths)-1))

    path_test_pred_vessels_tumors = os.path.join(folder_prediction_vessels_tumors, image_path)
    path_test_gt_vessels_tumors   = os.path.join(folder_vessels_tumors_masks, image_path)

    pred_vessels_tumors_before = nib.load(path_test_pred_vessels_tumors)
    pred_vessels_tumors_before = pred_vessels_tumors_before.get_data()
    gt_vessels_tumors          = nib.load(path_test_gt_vessels_tumors)
    gt_vessels_tumors          = gt_vessels_tumors.get_data()

    if do_class_report:
        cr_before = classification_report(gt_vessels_tumors.flatten(), pred_vessels_tumors_before.flatten(),
                                          labels=labels, target_names=target_names, output_dict=True)

        cr_before_list.append(cr_before)


        print('Classification Report Before:\n{}'.format(cr_before))

    # ACCURACY
    accuracy_before_cumul += (pred_vessels_tumors_before == gt_vessels_tumors).sum() / float(gt_vessels_tumors.size)

    # DICES
    for cls in range(0, NUM_CLASSES):
        outputs_cls_before = (pred_vessels_tumors_before == cls)
        labels_cls         = (gt_vessels_tumors == cls)
        intersect_cls_before = (np.logical_and(outputs_cls_before, labels_cls)).sum()
        union_cls_before = np.sum(outputs_cls_before) + np.sum(labels_cls)

        intersect_before[cls] += intersect_cls_before
        union_before[cls] += union_cls_before


accuracy_before = accuracy_before_cumul / NUM_IMAGES

print('Accuracy     = {}'.format(accuracy_before))

dice_cls_before = np.zeros(NUM_CLASSES)
for cls in range(0, NUM_CLASSES):
    dice_cls_before [cls] = (2 * intersect_before[cls] + eps) / (union_before[cls] + eps)


avg_dice_before = np.mean(dice_cls_before)
print('Average Dice = {}'.format(avg_dice_before))


metrics = {
    'Accuracy' : accuracy_before,
    'AvgDice'  : avg_dice_before,
    'DiceCls'  : list(dice_cls_before),
}

import json
json_path = os.path.join(folder_dataset, 'metrics_{}.json'.format("vessels_tumors_net"))
with open(json_path, 'w') as fp:
    json.dump(metrics, fp)