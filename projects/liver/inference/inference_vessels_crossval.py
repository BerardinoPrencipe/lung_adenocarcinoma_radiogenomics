import numpy as np
import torch
import nibabel as nib
import os
import sys
import platform
from utils_calc import normalize_data, normalize_data_old, get_mcc
from projects.liver.util.inference import perform_inference_volumetric_image
from projects.liver.train.config import window_hu
from semseg.models.vnet_v2 import VXNet
import medpy.metric.binary as mmb
from sklearn.metrics import confusion_matrix, matthews_corrcoef


current_path_abs = os.path.abspath('.')
sys.path.append(current_path_abs)
print('{} appended to sys!'.format(current_path_abs))

from utils_calc import normalize_data, get_patient_id
from projects.liver.train.config import window_hu

# Use local path or absolute
if 'Ubuntu' in platform.system() or 'Linux' in platform.system():
    isLinux = True
else:
    isLinux = False

if torch.cuda.device_count() > 1:
    cuda_dev = torch.device('cuda:1')
else:
    cuda_dev = torch.device('cuda')

### variables ###
do_mask_with_liver = True
cross_val_steps = 4
trainval_images = 16
test_img = "02"
exclude_list = ["10","12","20"]

dataset_folder = os.path.join(current_path_abs, 'datasets/ircadb')
source_folder = os.path.join(dataset_folder, 'nii')

source_subolders = os.listdir(source_folder)
source_subolders.sort()
print('Source Folder  = {}'.format(source_folder))
print('Source SubDirs = {}'.format(source_subolders))

LIVER_CLASS = (1,)
VESSELS_CLASS = (2, 3)
ARTERY_CLASS = (4,)

# rand_perm_val = np.random.permutation(trainval_images)
rand_perm_val = [ 7, 12,  8, 11,
                  1,  4, 10,  9,
                  5,  0,  3, 13,
                 14,  6,  2, 15]
print("Rand Perm Val = ", rand_perm_val)

models_paths_list = [
    "dummy1",
    "dummy2",
    "dummy3",
    "dummy4",
    "dummy5",
]

metrics = {key : dict() for key in models_paths_list}

# validation list
for idx_crossval in range(cross_val_steps+1):
    if idx_crossval == cross_val_steps:
        val_list = [test_img,]
    else:
        rand_perm_val_list = ["{:02d}".format(idx+1) for idx in rand_perm_val]
        val_list = rand_perm_val_list[idx_crossval*(trainval_images//cross_val_steps):
                                      (idx_crossval+1)*(trainval_images//cross_val_steps)]
        val_list.append(test_img)
    print("Iter ", idx_crossval)
    print("Val List = ", val_list)

    path_net = models_paths_list[idx_crossval]
    print("Using Model with path = ", path_net)
    net = torch.load(path_net)

    # Initializing arrays for metrics
    ious = np.zeros(len(val_list))
    precisions = np.zeros(len(val_list))
    recalls = np.zeros(len(val_list))
    dices = np.zeros(len(val_list))
    rvds = np.zeros(len(val_list))
    assds = np.zeros(len(val_list))
    hds = np.zeros(len(val_list))

    accs = np.zeros(len(val_list))
    senss = np.zeros(len(val_list))
    specs = np.zeros(len(val_list))

    mccs = np.zeros(len(val_list))
    tps, tns, fps, fns = 0, 0, 0, 0

    for idx, subfolder_source in enumerate(source_subolders):
        print('Index {} on {}'.format(idx, len(os.listdir(source_folder)) - 1))

        image_folder = os.path.join(source_folder, subfolder_source, 'image')
        mask_folder = os.path.join(source_folder, subfolder_source, 'mask')

        image_filename = os.path.join(image_folder, 'image.nii')
        mask_filename = os.path.join(mask_folder, 'mask.nii')

        # load file
        image_data = nib.load(image_filename)
        mask_data = nib.load(mask_filename)
        voxel_spacing = mask_data.header.get_zooms()

        # convert to numpy
        mask_data = mask_data.get_data()
        mask_data_hv = (mask_data == VESSELS_CLASS[0]).astype(np.uint8)
        mask_data_pv = (mask_data == VESSELS_CLASS[1]).astype(np.uint8)
        mask_data = np.logical_or(mask_data_hv, mask_data_pv)
        image_data_no_norm = image_data.get_data()
        image_data_norm = normalize_data(image_data_no_norm, window_hu)

        # transpose so the z-axis (slices) are the first dimension
        image_data_norm = np.transpose(image_data_norm, (2, 0, 1))
        image_data_no_norm = np.transpose(image_data_no_norm, (2, 0, 1))
        mask_data = np.transpose(mask_data, (2, 0, 1))

        if do_mask_with_liver:
            # Load Liver Mask
            mask_liver_filename = os.path.join(mask_folder, 'liver_closed.nii')
            mask_liver_data = nib.load(mask_liver_filename)
            mask_liver_data = mask_liver_data.get_data()
            mask_liver_data = np.transpose(mask_liver_data, (2, 0, 1))

            # Mask Image and Label with Liver Mask
            i_dtype = image_data_norm.dtype
            m_dtype = mask_data.dtype
            image_data_norm = (image_data_norm * mask_liver_data).astype(i_dtype)
            mask_data = (mask_data * mask_liver_data).astype(m_dtype)

        output = perform_inference_volumetric_image(net, image_data_norm,
                                                    context=2, do_round=True, cuda_dev=cuda_dev)


        ## Metrics
        iou = mmb.jc(output, mask_data)
        dice = mmb.dc(output, mask_data)
        prec = mmb.precision(output, mask_data)
        recall = mmb.recall(output, mask_data)
        rvd = mmb.ravd(output, mask_data)
        assd = mmb.assd(output, mask_data, voxelspacing=voxel_spacing)
        hd = mmb.hd(output, mask_data, voxelspacing=voxel_spacing)

        print('Patient   = ', subfolder_source)
        print('\nVolumetric Overlap Metrics')
        print('IoU       = ', iou)
        print('Dice      = ', dice)
        print('Precision = ', prec)
        print('Recall    = ', recall)
        print('RVD       = ', rvd)
        print('ASSD      = ', assd)
        print('HD        = ', hd)

        ious[idx] = iou
        precisions[idx] = prec
        recalls[idx] = recall
        dices[idx] = dice
        rvds[idx] = rvd
        assds[idx] = assd
        hds[idx] = hd

        # CONFUSION MATRIX
        tn, fp, fn, tp = confusion_matrix(y_true=mask_data.flatten(), y_pred=output.flatten()).ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        sens = tp / (tp + fn)  # recall
        spec = tn / (tn + fp)
        # mcc = get_mcc(tp=tp, tn=tn, fp=fp, fn=fn)
        mcc = matthews_corrcoef(y_true=mask_data.flatten(), y_pred=output.flatten())

        tps += tp
        fps += fp
        fns += fn
        tns += tn

        accs[idx] = acc
        senss[idx] = sens
        specs[idx] = spec
        mccs[idx] = mcc

        print('\nConfusion Matrix Metrics')
        print('Accuracy  = {}'.format(acc))
        print('Sens (Re) = {}'.format(sens))
        print('Specif    = {}'.format(spec))
        print('MCC       = {}'.format(mcc))

    data = {
        'IoU': list(ious),
        'Precision': list(precisions),
        'Recall': list(recalls),
        'Dice': list(dices),
        'RVD': list(rvds),
        'ASSD': list(assds),
        'MSSD': list(hds),

        'Acc': list(accs),
        'Sens': list(senss),
        'Spec': list(specs),

        'Mcc': list(mccs),

        'TP': int(tps),
        'FP': int(fps),
        'FN': int(fns),
        'TN': int(tns),
    }

    metrics[path_net] = data

import json
json_path = os.path.join(current_path_abs, 'datasets/ircadb_crossval_metrics.json')
print('JSON Path = {}'.format(json_path))

with open(json_path, 'w') as f:
    json.dump(metrics, f)