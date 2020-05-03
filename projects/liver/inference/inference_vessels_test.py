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

path_net_vessels_tumors = 'logs/vessels_tumors/model_25D__2020-02-20__06_53_17.pht'

if model_to_use == "ircadb":
    # path_net = 'logs/vessels/model_25D__2020-01-15__08_28_39.pht'
    path_net = 'logs/vessels/model_25D__2020-03-12__10_37_59.pht'
elif model_to_use == "s_multi":
    path_net = 'logs/vessels_scardapane/model_25D__2020-03-27__07_11_38.pht'
    do_round = False
    do_argmax = True
elif model_to_use == "s_single":
    path_net = 'logs/vessels_scardapane_one_class/model_25D__2020-03-28__04_43_26.pht'
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

    continue

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

for idx, folder_patient_test in enumerate(folders_patients_test):
    print('Starting iter {} on {}'.format(idx+1,len(folders_patients_test)))
    print('Processing ', folder_patient_test)
    path_test_pred = os.path.join(folder_test_pred, folder_patient_test + ".nii.gz")
    path_test_gt = os.path.join(folder_test_images, folder_patient_test, "mask.nii.gz")

    gt_vessels_mask = nib.load(path_test_gt)
    gt_vessels_mask = gt_vessels_mask.get_data()
    gt_vessels_mask = 1*(gt_vessels_mask>0)
    output = nib.load(path_test_pred)
    output = output.get_data()

    dice = mmb.dc(output, gt_vessels_mask)
    prec = mmb.precision(output, gt_vessels_mask)
    recall = mmb.recall(output, gt_vessels_mask)

    tn, fp, fn, tp = confusion_matrix(y_true=gt_vessels_mask.flatten(), y_pred=output.flatten()).ravel()
    acc  = (tp+tn) / (tp+tn+fp+fn)
    spec = tn / (tn+fp)

    accs.append(acc)
    specs.append(spec)
    dices.append(dice)
    precs.append(prec)
    recalls.append(recall)

avg_dice   = np.mean(dices)
avg_acc    = np.mean(accs)
avg_recall = np.mean(recalls)
avg_spec   = np.mean(specs)
avg_prec   = np.mean(precs)

print('Average Dice      = {}'.format(avg_dice))
print('Average Accuracy  = {}'.format(avg_acc))
print('Average Precision = {}'.format(avg_prec))
print('Average Recall    = {}'.format(avg_recall))
print('Average Specif    = {}'.format(avg_spec))