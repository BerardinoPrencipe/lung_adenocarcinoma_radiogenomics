import numpy as np
import torch
import nibabel as nib
import os
import sys
import platform
import json
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
cross_val_steps = 3
trainval_images = 41

dataset_folder = os.path.join(current_path_abs, 'datasets/Task09_Spleen')
source_folder = os.path.join(dataset_folder, 'imagesTr')
labels_folder = os.path.join(dataset_folder, 'labelsTr')

source_images = os.listdir(source_folder)
source_images.sort()
print('Source Folder  = {}'.format(source_folder))
print('Source Images  = {}'.format(source_images))


# rand_perm_val = np.random.permutation(trainval_images)
rand_perm_val = ['008', '014', '010', '027', '029', '006', '017', '019', '063', '002',
                 '024', '044', '046', '041', '022', '016', '052', '033', '026', '049',
                 '056', '047', '021', '009', '018', '025', '060', '059', '032', '045',
                 '053', '040', '028', '031', '062', '061', '013', '038', '020', '012', '003']
print("Rand Perm Val = ", rand_perm_val)

# betas = ['09', '07', '05']
betas = ['07',]
"""
models_paths_list_09 = [
    "logs/spleen_crossval_00/model_25D__2020-11-10__06_14_56.pht",
    "logs/spleen_crossval_01/model_25D__2020-11-10__20_50_16.pht",
    "logs/spleen_crossval_02/model_25D__2020-11-11__10_54_35.pht",
    "logs/spleen_crossval_03/model_25D__2020-11-12__01_31_12.pht",,
]
"""
models_paths_list_07 = [
    "logs/spleen_crossval_00/model_25D__2020-11-10__06_14_56.pht",
    "logs/spleen_crossval_01/model_25D__2020-11-10__20_50_16.pht",
    "logs/spleen_crossval_02/model_25D__2020-11-11__10_54_35.pht",
    "logs/spleen_crossval_03/model_25D__2020-11-12__01_31_12.pht",
]
"""
models_paths_list_05 = [
    "logs/spleen_crossval_00/model_25D__2020-11-10__06_14_56.pht",
    "logs/spleen_crossval_01/model_25D__2020-11-10__20_50_16.pht",
    "logs/spleen_crossval_02/model_25D__2020-11-11__10_54_35.pht",
    "logs/spleen_crossval_03/model_25D__2020-11-12__01_31_12.pht",
]
"""
"""
models_paths_list_of_lists = [
    models_paths_list_09,
    models_paths_list_07,
    models_paths_list_05
]
"""
models_paths_list_of_lists = [
    models_paths_list_07,
]

for beta, models_paths_list in zip(betas, models_paths_list_of_lists):

    metrics = {key : dict() for key in models_paths_list}

    # validation list
    for idx_crossval in range(cross_val_steps+1):
        rand_perm_val_list = rand_perm_val
        val_list = rand_perm_val_list[idx_crossval*(trainval_images//cross_val_steps):
                                      (idx_crossval+1)*(trainval_images//cross_val_steps)]
        print("Iter ", idx_crossval)
        print("Val List = ", val_list)

        path_net = models_paths_list[idx_crossval]
        print("Using Model with path = ", path_net)
        net = torch.load(path_net)
        net = net.cuda(cuda_dev)
        net.eval()

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

        idx_val_in_dataset = -1

        for idx, image_id in enumerate(source_images):
            print('Index {} on {}'.format(idx+1, len(os.listdir(source_folder))))
            id_patient = image_id[7:10]
            if id_patient not in val_list:
                print(id_patient, "is not in val list")
                continue
            else:
                print(id_patient, "is in val list")
                idx_val_in_dataset += 1

            image_filename = os.path.join(source_folder, image_id)
            label_filename = os.path.join(labels_folder, image_id)

            # load file
            image_data = nib.load(image_filename)
            mask_data = nib.load(label_filename)
            voxel_spacing = mask_data.header.get_zooms()

            # convert to numpy
            mask_data = mask_data.get_data().astype(np.uint8)
            image_data_no_norm = image_data.get_data()
            image_data_norm = normalize_data(image_data_no_norm, window_hu)

            # transpose so the z-axis (slices) are the first dimension
            image_data_norm = np.transpose(image_data_norm, (2, 0, 1))
            image_data_no_norm = np.transpose(image_data_no_norm, (2, 0, 1))
            mask_data = np.transpose(mask_data, (2, 0, 1))

            output = perform_inference_volumetric_image(net, image_data_norm,
                                                        context=2, do_round=True, cuda_dev=cuda_dev)

            # Metrics
            iou = mmb.jc(output, mask_data)
            dice = mmb.dc(output, mask_data)
            prec = mmb.precision(output, mask_data)
            recall = mmb.recall(output, mask_data)
            rvd = mmb.ravd(output, mask_data)
            assd = mmb.assd(output, mask_data, voxelspacing=voxel_spacing)
            hd = mmb.hd(output, mask_data, voxelspacing=voxel_spacing)

            print('Patient   = ', id_patient)
            print('\nVolumetric Overlap Metrics')
            print('IoU       = ', iou)
            print('Dice      = ', dice)
            print('Precision = ', prec)
            print('Recall    = ', recall)
            print('RVD       = ', rvd)
            print('ASSD      = ', assd)
            print('HD        = ', hd)

            ious[idx_val_in_dataset] = iou
            precisions[idx_val_in_dataset] = prec
            recalls[idx_val_in_dataset] = recall
            dices[idx_val_in_dataset] = dice
            rvds[idx_val_in_dataset] = rvd
            assds[idx_val_in_dataset] = assd
            hds[idx_val_in_dataset] = hd

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

            accs[idx_val_in_dataset] = acc
            senss[idx_val_in_dataset] = sens
            specs[idx_val_in_dataset] = spec
            mccs[idx_val_in_dataset] = mcc

            print('\nConfusion Matrix Metrics')
            print('Accuracy  = {}'.format(acc))
            print('Sens (Re) = {}'.format(sens))
            print('Specif    = {}'.format(spec))
            print('MCC       = {}'.format(mcc))

        ious_avg = np.mean(ious)
        ious_std = np.std(ious, ddof=1)
        dices_avg = np.mean(dices)
        dices_std = np.std(dices, ddof=1)
        rvds_avg = np.mean(rvds)
        rvds_std = np.std(rvds, ddof=1)
        assds_avg = np.mean(assds)
        assds_std = np.std(assds, ddof=1)
        hds_avg = np.mean(hds)
        hds_std = np.std(hds, ddof=1)

        print("IoU  = {:.4f} +/- {:.4f}".format(ious_avg, ious_std))
        print("Dice = {:.4f} +/- {:.4f}".format(dices_avg, dices_std))
        print("RVD  = {:.4f} +/- {:.4f}".format(rvds_avg, rvds_std))
        print("ASSD = {:.4f} +/- {:.4f}".format(assds_avg, assds_std))
        print("HD   = {:.4f} +/- {:.4f}".format(hds_avg, hds_std))

        data = {
            'AvgIou' : ious_avg,
            'StdIou' : ious_std,
            'AvgDice': dices_avg,
            'StdDice': dices_std,
            'AvgRvd' : rvds_avg,
            'StdRvd' : rvds_std,
            'AvgASSD': assds_avg,
            'StdASSD': assds_std,
            'AvgHD'  : hds_avg,
            'StdHD'  : hds_std,

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

    json_path = os.path.join(current_path_abs, 'datasets/spleen_crossval_metrics_{}.json'.format(beta))
    print('JSON Path = {}'.format(json_path))

    with open(json_path, 'w') as f:
        json.dump(metrics, f)
