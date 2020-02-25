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

from utils_calc import normalize_data, normalize_data_old, get_mcc
from projects.liver.util.calc import correct_volume_slice_split, \
                                     erase_non_max_cc_segments
from projects.liver.util.inference import perform_inference_volumetric_image
from projects.liver.train.config import window_hu
from semseg.models.vnet_v2 import VXNet

do_mask_liver      = False
inference_vessels  = False
inference_segments = True
use_state_dict     = False

# Load net
if inference_segments:
    if use_state_dict:
        path_net_segments = os.path.join(current_path_abs, 'logs/segments/model_epoch_0040.pht')
        net_segments = VXNet(dropout=True, context=2, num_outs=9, no_softmax=False)
        net_segments.load_state_dict(torch.load(path_net_segments))
    else:
        path_net_segments = os.path.join(current_path_abs, 'logs/segments/model_25D__2020-02-19__07_13_36.pht')
        net_segments = torch.load(path_net_segments)
    print('Network Path Segments = {}'.format(path_net_segments))

if inference_vessels:
    path_net_vessels = os.path.join(current_path_abs, 'logs/vessels_tumors/model_25D__2020-02-20__06_53_17.pht')
    net_vessels = torch.load(path_net_vessels)
    print('Network Path Vessels  = {}'.format(path_net_vessels))

if torch.cuda.device_count() > 1:
    cuda_dev = torch.device('cuda:1')
else:
    cuda_dev = torch.device('cuda')
print('Device Count = {}, using CUDA Device = {}'.format(torch.cuda.device_count(), cuda_dev))

if inference_segments:
    net_segments = net_segments.cuda(cuda_dev)
    net_segments.eval()

if inference_vessels:
    net_vessels = net_vessels.cuda(cuda_dev)
    net_vessels.eval()

folder_dataset        = 'datasets/LiverDecathlon/nii/images'
folder_liver_masks    = 'datasets/LiverDecathlon/nii/labels_liver'
folder_segments_masks = 'datasets/LiverDecathlon/nii/labels_segments'
folder_prediction_segments_before = 'datasets/LiverDecathlon/nii/pred_segments'
folder_prediction_segments_after  = 'datasets/LiverDecathlon/nii/pred_segments_after'
folder_prediction_vessels  = 'datasets/LiverDecathlon/nii/pred_vessels'

if not os.path.exists(folder_prediction_segments_before):
    os.makedirs(folder_prediction_segments_before)

if not os.path.exists(folder_prediction_segments_after):
    os.makedirs(folder_prediction_segments_after)

if not os.path.exists(folder_prediction_vessels):
    os.makedirs(folder_prediction_vessels)

image_paths = []
for idx in range(1,50):
    image_path = 'hepaticvessel_{:03d}.nii.gz'.format(idx)
    if image_path in os.listdir(folder_segments_masks):
        image_paths.append(image_path)
    else:
        print('Image Path {} not in Ground Truth Segments'.format(image_path))
# idxs_train = [70, 80]
# train_image_paths = ['hepaticvessel_{:03d}.nii.gz'.format(idx_train) for idx_train in idxs_train]
# image_paths += train_image_paths

# Start iteration over val set
for idx, image_path in enumerate(image_paths):

    path_test_image = os.path.join(folder_dataset, image_path)
    path_test_pred_segments_before = os.path.join(folder_prediction_segments_before, image_path)
    path_test_pred_segments_after  = os.path.join(folder_prediction_segments_after, image_path)
    path_test_pred_vessels  = os.path.join(folder_prediction_vessels, image_path)
    path_test_liver = os.path.join(folder_liver_masks, image_path)

    # load file
    data = nib.load(path_test_image)
    # save affine
    input_aff = data.affine
    # convert to numpy
    data = data.get_data()

    if do_mask_liver:
        data_liver = nib.load(path_test_liver)
        data_liver = data_liver.get_data()
        data[data_liver == 0] = 0

    # normalize data
    data = normalize_data(data, window_hu)

    # transpose so the z-axis (slices) are the first dimension
    data = np.transpose(data, (2, 0, 1))

    # CNN
    if inference_segments:
        output_segments_before = perform_inference_volumetric_image(net_segments, data, context=2,
                                                                    do_round=False, do_argmax=True, cuda_dev=cuda_dev)
        print('Shape before correction: {}'.format(output_segments_before.shape))
        output_segments = erase_non_max_cc_segments(output_segments_before)
        output_segments = correct_volume_slice_split(output_segments)
        print('Shape after  correction: {}'.format(output_segments.shape))
        output_segments = np.transpose(output_segments, (1, 2, 0)).astype(np.uint8)

        non_zero_elements_segments = np.count_nonzero(output_segments)
        print("Non-Zero elements segments = {}".format(non_zero_elements_segments))

        output_segments_before = np.transpose(output_segments_before, (1, 2, 0)).astype(np.uint8)
        output_nib_segments_before = nib.Nifti1Image(output_segments_before, affine=input_aff)
        nib.save(output_nib_segments_before, path_test_pred_segments_before)

        output_nib_segments_after = nib.Nifti1Image(output_segments, affine=input_aff)
        nib.save(output_nib_segments_after, path_test_pred_segments_after)

    if inference_vessels:
        output_vessels = perform_inference_volumetric_image(net_vessels, data, context=2,
                                                            do_round=False, do_argmax=True, cuda_dev=cuda_dev)
        output_vessels = np.transpose(output_vessels, (1,2,0)).astype(np.uint8)

        non_zero_elements_vessels = np.count_nonzero(output_vessels)
        print("Non-Zero elements vessels = {}".format(non_zero_elements_vessels))

        output_nib_vessels = nib.Nifti1Image(output_vessels, affine=input_aff)
        nib.save(output_nib_vessels, path_test_pred_vessels)

    print("Index {} on {}.\n"
          "Segments Image saved in {} and {} \n"
          "Vessels Image saved  in {}".format(idx, len(image_paths) - 1,
                                              path_test_pred_segments_before, path_test_pred_segments_after,
                                              path_test_pred_vessels))


#########################
### CALCULATE METRICS ###
#########################

from sklearn.metrics import classification_report

target_names = ['b'] + ['s{:d}'.format(idx+1) for idx in range(8)]
labels = [idx for idx in range(9)]

cr_before_list = []
cr_after_list  = []

# Start iteration over val set
for idx, image_path in enumerate(image_paths):
    print('Starting Iter {} on {}...'.format(idx, len(image_paths)-1))

    path_test_pred_segments_before = os.path.join(folder_prediction_segments_before, image_path)
    path_test_pred_segments_after  = os.path.join(folder_prediction_segments_after, image_path)
    path_test_gt_segments          = os.path.join(folder_segments_masks, image_path)

    pred_segments_before = nib.load(path_test_pred_segments_before)
    pred_segments_before = pred_segments_before.get_data()
    pred_segments_after  = nib.load(path_test_pred_segments_after)
    pred_segments_after  = pred_segments_after.get_data()
    gt_segments          = nib.load(path_test_gt_segments)
    gt_segments          = gt_segments.get_data()

    cr_before = classification_report(gt_segments.flatten(), pred_segments_before.flatten(),
                                      labels=labels, target_names=target_names, output_dict=True)
    cr_after  = classification_report(gt_segments.flatten(), pred_segments_after.flatten(),
                                      labels=labels, target_names=target_names, output_dict=True)

    cr_before_list.append(cr_before)
    cr_after_list.append(cr_after)

    print('Classification Report Before:\n{}'.format(cr_before))
    print('Classification Report After :\n{}'.format(cr_after))