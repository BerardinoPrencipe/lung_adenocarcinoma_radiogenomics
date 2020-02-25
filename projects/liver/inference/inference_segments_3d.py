import os
import torch
import sys
import numpy as np
import nibabel as nib

current_path_abs = os.path.abspath('.')
sys.path.append(current_path_abs)
print('{} appended to sys!'.format(current_path_abs))

from utils_calc import normalize_data, normalize_data_old, \
                       perform_inference_volumetric_image, perform_inference_volumetric_image_3d, get_mcc
from projects.liver.train.config import window_hu
from semseg.models.vnet3d import VXNet3D


path_net = os.path.join(current_path_abs, 'logs/segments/model_3D_epoch_0466.pht')
net = VXNet3D(dropout=True,num_outs=9,no_softmax=False)
net.load_state_dict(torch.load(path_net))

if torch.cuda.device_count() > 1:
    cuda_dev = torch.device('cuda:1')
else:
    cuda_dev = torch.device('cuda')
print('Device Count = {}, using CUDA Device = {}'.format(torch.cuda.device_count(), cuda_dev))

net = net.cuda(cuda_dev)
net.eval()

folder_dataset             = os.path.join(current_path_abs, 'datasets/LiverDecathlon/nii/images')
folder_prediction_segments = os.path.join(current_path_abs, 'datasets/LiverDecathlon/nii/pred_segments_3d')

if not os.path.exists(folder_prediction_segments):
    os.makedirs(folder_prediction_segments)

path_test_preds = []
image_paths = ['hepaticvessel_{:03d}.nii.gz'.format(idx) for idx in [1, 71]]

# Start iteration over val set
for idx, image_path in enumerate(image_paths):

    path_test_image = os.path.join(folder_dataset, image_path)
    path_test_pred_segments = os.path.join(folder_prediction_segments, image_path)
    path_test_preds.append(path_test_pred_segments)

    # load file
    data = nib.load(path_test_image)
    # save affine
    input_aff = data.affine
    # convert to numpy
    data = data.get_data()

    # normalize data
    data = normalize_data(data, window_hu)

    # transpose so the z-axis (slices) are the first dimension
    data = np.transpose(data, (2, 0, 1))

    # CNN
    output_segments = perform_inference_volumetric_image_3d(net, data, depth=16,
                                                         do_round=False, do_argmax=True, cuda_dev=cuda_dev)
    output_segments = np.transpose(output_segments, (1, 2, 0)).astype(np.uint8)

    non_zero_elements_segments = np.count_nonzero(output_segments)
    print("Non-Zero elements segments = {}".format(non_zero_elements_segments))

    output_nib_segments = nib.Nifti1Image(output_segments, affine=input_aff)
    nib.save(output_nib_segments, path_test_pred_segments)

    print("Index {} on {}.\n"
          "Segments Image saved in {}\n".format(idx, len(image_paths) - 1, path_test_pred_segments ))