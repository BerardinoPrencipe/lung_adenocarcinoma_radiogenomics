import os
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import torch
import medpy.metric.binary as mmb
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from utils_calc import normalize_data
from projects.liver.util.inference import perform_inference_volumetric_image
from projects.liver.train.config import window_hu
import json

folder_logs = 'logs'

use_in_lab = False
if use_in_lab:
    folder_test_dataset = 'E:/Datasets/LiverScardapane'
else:
    folder_test_dataset = 'H:/Datasets/Liver/LiverScardapane'
folder_test_images = os.path.join(folder_test_dataset, 'ct_scans')
folders_patients_test = os.listdir(folder_test_images)
folders_patients_test = [folder for folder in folders_patients_test
                         if os.path.isdir(os.path.join(folder_test_images, folder))]

folders_patients_test = folders_patients_test[0:4]

path_net = None
do_round = True
do_argmax = False
model_to_use = "ircadb"
folder_test_pred = os.path.join(folder_test_dataset, model_to_use)
if not os.path.exists(folder_test_pred):
    os.makedirs(folder_test_pred)

# path_net_vessels_tumors = 'logs/vessels_tumors/model_25D__2020-02-20__06_53_17.pht'
# path_net = 'logs/vessels/model_25D__2020-01-15__08_28_39.pht'

# MODELS WITH DROPOUT
# Dice / Tversky 0.7 / Tversky 0.9
models_paths = [
    'logs/vessels/model_25D__2020-03-12__10_37_59.pht',
    'logs/vessels/model_25D__2020-05-04__09_55_18.pht',
    'logs/vessels/model_25D__2020-05-07__09_09_33.pht'
]

# MODELS WITHOUT DROPOUT
# Dice / Tversky 0.7 / Tversky 0.9
""" 
model_paths = [
    'logs/vessels/model_25D__2020-06-18__12_54_58.pht',
    'logs/vessels/model_25D__2020-06-19__10_22_00.pht',
    '
]
"""

for path_net in models_paths:

    print("Path Net = ", path_net)

    # Load net
    net = torch.load(path_net)
    cuda_device = torch.device('cuda:0')
    net.to(cuda_device)

    sizes = []
    spacings = []

    #%% Start iteration over val set
    for idx, folder_patient_test in enumerate(folders_patients_test):
        print('Starting iter {} on {}'.format(idx+1,len(folders_patients_test)))
        print('Processing ', folder_patient_test)
        path_test_pred = os.path.join(folder_test_pred, folder_patient_test + ".nii.gz")
        path_test_folder = os.path.join(folder_test_images, folder_patient_test)

        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(path_test_folder)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        size = image.GetSize()
        sizes.append(size)
        print("Image size   : ", size[0], size[1], size[2])
        spacing = image.GetSpacing()
        spacings.append(spacing)
        print("Image spacing: ", spacing)

        image_data = sitk.GetArrayFromImage(image)
        # normalize data
        data = normalize_data(image_data, window_hu)

        # transpose so the z-axis (slices) are the first dimension
        data = np.transpose(data, (0, 2, 1))
        # CNN
        output = perform_inference_volumetric_image(net, data, context=2,
                                                    do_round=do_round, do_argmax=do_argmax,
                                                    cuda_dev=cuda_device)
        output = np.transpose(output, (1, 2, 0)).astype(np.uint8)

        n_nonzero = np.count_nonzero(output)
        print("Non-zero elements = ", n_nonzero)

        output_nib_pre = nib.Nifti1Image(output, affine=None)
        nib.save(output_nib_pre, path_test_pred)


    #%% Compute metrics
    dices = []
    precs = []
    recalls = []
    accs = []
    specs = []
    assds = []
    hds = []

    for idx, folder_patient_test in enumerate(folders_patients_test):
        print('Starting iter {} on {}'.format(idx+1,len(folders_patients_test)))
        print('Processing ', folder_patient_test)
        path_test_pred = os.path.join(folder_test_pred, folder_patient_test + ".nii.gz")
        path_test_gt = os.path.join(folder_test_images, folder_patient_test, "mask.nii.gz")

        gt_vessels_mask = nib.load(path_test_gt)
        voxel_spacing = gt_vessels_mask.header.get_zooms()
        gt_vessels_mask = gt_vessels_mask.get_data()
        gt_vessels_mask = 1*(gt_vessels_mask>0)
        output = nib.load(path_test_pred)
        output = output.get_data()

        dice = mmb.dc(output, gt_vessels_mask)
        prec = mmb.precision(output, gt_vessels_mask)
        recall = mmb.recall(output, gt_vessels_mask)

        assd = mmb.assd(output, gt_vessels_mask, voxelspacing=voxel_spacing)
        hd = mmb.hd(output, gt_vessels_mask, voxelspacing=voxel_spacing)

        tn, fp, fn, tp = confusion_matrix(y_true=gt_vessels_mask.flatten(), y_pred=output.flatten()).ravel()
        acc  = (tp+tn) / (tp+tn+fp+fn)
        spec = tn / (tn+fp)

        accs.append(acc)
        specs.append(spec)
        dices.append(dice)
        precs.append(prec)
        recalls.append(recall)
        assds.append(assd)
        hds.append(hd)

    avg_dice   = np.mean(dices)
    avg_acc    = np.mean(accs)
    avg_recall = np.mean(recalls)
    avg_spec   = np.mean(specs)
    avg_prec   = np.mean(precs)
    avg_assd   = np.mean(assds)
    avg_hd     = np.mean(hds)

    std_dice   = np.std(dices)
    std_acc    = np.std(accs)
    std_recall = np.std(recalls)
    std_spec   = np.std(specs)
    std_prec   = np.std(precs)
    std_assd   = np.std(assds)
    std_hd     = np.std(hds)

    print('Avg +/- Std Dice      = {:.2f} +/- {:.2f}'.format(avg_dice * 100, std_dice * 100))
    print('Avg +/- Std Accuracy  = {:.2f} +/- {:.2f}'.format(avg_acc * 100, std_acc * 100))
    print('Avg +/- Std Precision = {:.2f} +/- {:.2f}'.format(avg_prec * 100, std_prec * 100))
    print('Avg +/- Std Recall    = {:.2f} +/- {:.2f}'.format(avg_recall * 100, std_recall * 100))
    print('Avg +/- Std Specif    = {:.2f} +/- {:.2f}'.format(avg_spec * 100, std_spec * 100))
    print('Avg +/- Std ASSD      = {:.2f} +/- {:.2f}'.format(avg_assd, std_assd))
    print('Avg +/- Std HD        = {:.2f} +/- {:.2f}'.format(avg_hd, std_hd))

    metrics = {
        'AvgDice'       : avg_dice,
        'AvgAcc'        : avg_acc,
        'AvgPrecision'  : avg_prec,
        'AvgRecall'     : avg_recall,
        'AvgSpecif'     : avg_spec,
        'AvgASSD'       : avg_assd,
        'AvgHD'         : avg_hd,

        'StdDice'       : std_dice,
        'StdAcc'        : std_acc,
        'StdPrecision'  : std_prec,
        'StdRecall'     : std_recall,
        'StdSpecif'     : std_spec,
        'StdASSD'       : std_assd,
        'StdHD'         : std_hd,

        'Dices'         : dices,
        'Accs'          : accs,
        'Recalls'       : recalls,
        'Precisions'    : precs,
        'Specifs'       : specs,
        'ASSDs'         : assds,
        'HDs'           : hds
    }

    json_path = os.path.join(folder_logs, 'metrics_{}.json'.format(path_net.split('/')[-1]))
    with open(json_path, 'w') as fp:
        json.dump(metrics, fp)