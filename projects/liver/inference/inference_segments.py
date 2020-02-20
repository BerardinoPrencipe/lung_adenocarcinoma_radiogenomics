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

do_mask_liver  = False
use_state_dict = False

# Load net
if use_state_dict:
    path_net = os.path.join(current_path_abs, 'logs/segments/model_epoch_0040.pht')
    net = VXNet(dropout=True,context=2,num_outs=9,no_softmax=False)
    net.load_state_dict(torch.load(path_net))
else:
    path_net = os.path.join(current_path_abs, 'logs/segments/model_25D__2020-02-19__07_13_36.pht')
    net = torch.load(path_net)

path_net_vessels = os.path.join(current_path_abs, 'logs/vessels_tumors/model_25D__2020-02-20__06_53_17.pht')
net_vessels = torch.load(path_net_vessels)

if torch.cuda.device_count() > 1:
    cuda_dev = torch.device('cuda:1')
else:
    cuda_dev = torch.device('cuda')
print('Device Count = {}, using CUDA Device = {}'.format(torch.cuda.device_count(), cuda_dev))

net = net.cuda(cuda_dev)
net.eval()

net_vessels = net_vessels.cuda(cuda_dev)
net_vessels.eval()

print('Network Path Segments = {}'.format(path_net))
print('Network Path Vessels  = {}'.format(path_net_vessels))


path_test_preds = list()

folder_dataset     = 'E:/Datasets/LiverDecathlon/nii/images'
folder_liver_masks = 'E:/Datasets/LiverDecathlon/nii/labels_liver'
folder_prediction_segments = 'E:/Datasets/LiverDecathlon/nii/pred_segments'
folder_prediction_vessels  = 'E:/Datasets/LiverDecathlon/nii/pred_vessels'

if not os.path.exists(folder_prediction_segments):
    os.makedirs(folder_prediction_segments)

if not os.path.exists(folder_prediction_vessels):
    os.makedirs(folder_prediction_vessels)

image_paths = []
for idx in range(1,6):
    image_path = 'hepaticvessel_{:03d}.nii.gz'.format(idx)
    image_paths.append(image_path)
idxs_train = [70, 80]
train_image_paths = ['hepaticvessel_{:03d}.nii.gz'.format(idx_train) for idx_train in idxs_train]
image_paths += train_image_paths

# Start iteration over val set
for idx, image_path in enumerate(image_paths):

    path_test_image = os.path.join(folder_dataset, image_path)
    path_test_pred_segments = os.path.join(folder_prediction_segments, image_path)
    path_test_pred_vessels  = os.path.join(folder_prediction_vessels, image_path)
    path_test_liver = os.path.join(folder_liver_masks, image_path)
    path_test_preds.append(path_test_pred_segments)

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
    output_segments = perform_inference_volumetric_image(net, data, context=2,
                                                         do_round=False, do_argmax=True, cuda_dev=cuda_dev)
    output_segments = np.transpose(output_segments, (1, 2, 0)).astype(np.uint8)

    output_vessels = perform_inference_volumetric_image(net_vessels, data, context=2,
                                                        do_round=False, do_argmax=True, cuda_dev=cuda_dev)
    output_vessels = np.transpose(output_vessels, (1,2,0)).astype(np.uint8)

    non_zero_elements_segments = np.count_nonzero(output_segments)
    print("Non-Zero elements segments = {}".format(non_zero_elements_segments))

    non_zero_elements_vessels = np.count_nonzero(output_vessels)
    print("Non-Zero elements vessels = {}".format(non_zero_elements_vessels))

    output_nib_segments = nib.Nifti1Image(output_segments, affine=input_aff)
    nib.save(output_nib_segments, path_test_pred_segments)

    output_nib_vessels = nib.Nifti1Image(output_vessels, affine=input_aff)
    nib.save(output_nib_vessels, path_test_pred_vessels)
    print("Index {} on {}.\n"
          "Segments Image saved in {}\n"
          "Vessels Image saved  in {}".format(idx, len(image_paths) - 1,
                                              path_test_pred_segments, path_test_pred_vessels))