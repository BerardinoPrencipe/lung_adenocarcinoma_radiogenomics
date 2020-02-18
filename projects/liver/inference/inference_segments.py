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
    # path_net = os.path.join(current_path_abs, 'logs/segments/model_25D__2020-02-17__14_28_37.pht')
    # path_net = os.path.join(current_path_abs, 'logs/segments/model_25D__2020-02-17__16_40_34.pht')
    # path_net = os.path.join(current_path_abs, 'logs/segments/model_25D__2020-02-17__23_53_26.pht')
    # path_net = os.path.join(current_path_abs, 'logs/segments/model_25D__2020-02-18__13_11_43.pht')
    path_net = os.path.join(current_path_abs, 'logs/segments/model_25D__2020-02-18__15_32_47.pht')
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

folder_dataset     = 'E:/Datasets/LiverDecathlon/nii/images'
folder_liver_masks = 'E:/Datasets/LiverDecathlon/nii/labels_liver'
folder_prediction  = 'E:/Datasets/LiverDecathlon/nii/pred_segments'

if not os.path.exists(folder_prediction):
    os.makedirs(folder_prediction)

image_paths = []
for idx in range(1,6):
    image_path = 'hepaticvessel_{:03d}.nii.gz'.format(idx)
    image_paths.append(image_path)

# Start iteration over val set
for idx, image_path in enumerate(image_paths):

    path_test_image = os.path.join(folder_dataset, image_path)
    path_test_pred  = os.path.join(folder_prediction, image_path)
    path_test_liver = os.path.join(folder_liver_masks, image_path)
    path_test_preds.append(path_test_pred)

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
    output = perform_inference_volumetric_image(net, data, context=2,
                                                do_round=False, do_argmax=True, cuda_dev=cuda_dev)
    output = np.transpose(output, (1, 2, 0)).astype(np.uint8)

    non_zero_elements = np.count_nonzero(output)
    print("Non-Zero elements = {}".format(non_zero_elements))

    output_nib = nib.Nifti1Image(output, affine=input_aff)
    nib.save(output_nib, path_test_pred)
    print("Index {} on {}.\nImage saved in {}".format(idx, len(image_paths)-1, path_test_pred))