import os
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import torch
from utils import normalize_data, perform_inference_volumetric_image


val_list = ["{:02d}".format(idx) for idx in range(1,3)]
val_list.append("10")

folder_dataset = 'datasets/ircadb'
folders_patients = os.listdir(folder_dataset)
folders_patients_train = [folder for folder in folders_patients if not any(val_el in folder for val_el in val_list)]
folders_patients_valid = [folder for folder in folders_patients if any(val_el in folder for val_el in val_list)]

path_net = 'logs/vessels/model_25D__2020-01-15__08_28_39.pht'
# Load net
net = torch.load(path_net)

# Start iteration over val set
for folder_patient_valid in folders_patients_valid:

    path_test_image = os.path.join(folder_dataset, folder_patient_valid, 'image', 'image.nii')
    path_test_pred  = os.path.join(folder_dataset, folder_patient_valid, 'image', 'pred.nii')

    # load file
    data = nib.load(path_test_image)
    # save affine
    input_aff = data.affine
    # convert to numpy
    data = data.get_data()

    # normalize data
    # data = normalize_data(data, dmin=-150, dmax=350)

    # transpose so the z-axis (slices) are the first dimension
    data = np.transpose(data, (2, 0, 1))
    # CNN
    output = perform_inference_volumetric_image(net, data, context=2, do_round=True)
    output = np.transpose(output, (1, 2, 0)).astype(np.uint8)

    n_nonzero = np.count_nonzero(output)
    print("Non-zero elements = ", n_nonzero)

    output_nib_pre = nib.Nifti1Image(output, affine=input_aff)
    nib.save(output_nib_pre, path_test_pred)

# TODO: post processing
# Connected components labeling
# Get connected components
ccif = sitk.ConnectedComponentImageFilter()
sitk_output = sitk.GetImageFromArray(output)
conn_comps = ccif.Execute(sitk_output)
conn_comps_np = sitk.GetArrayFromImage(conn_comps)

unique_values = np.unique(conn_comps_np)
n_uniques = len(unique_values)
counter_uniques = np.zeros(n_uniques)
for i in range(1, max(unique_values) + 1):
    counter_uniques[i] = (conn_comps_np == i).sum()
biggest_region_value = np.argmax(counter_uniques)

# Morphological closing
vector_radius = (10,10,10)
kernel = sitk.sitkBall
output_uint8 = output.astype(np.uint8)
output_sitk = sitk.GetImageFromArray(output_uint8)
output_closed_sitk = sitk.BinaryMorphologicalClosing(output_sitk, vector_radius, kernel)
output_closed_np = sitk.GetArrayFromImage(output_closed_sitk)

conn_comps_closed = ccif.Execute(output_closed_sitk)
conn_comps_closed_np = sitk.GetArrayFromImage(conn_comps_closed)

# Morphological opening
vector_radius = (10,10,10)
kernel = sitk.sitkBall
# morph_open_out = sitk.BinaryMorphologicalOpening(conn_comps_closed, vector_radius, kernel)
morph_open_out = sitk.BinaryMorphologicalOpening(output_sitk, vector_radius, kernel)

conn_comps_opened = ccif.Execute(morph_open_out)
conn_comps_opened_np = sitk.GetArrayFromImage(conn_comps_opened)

#
unique_values = np.unique(conn_comps_opened_np)
n_uniques = len(unique_values)
counter_uniques = np.zeros(n_uniques)
for i in range(1, max(unique_values) + 1):
    counter_uniques[i] = (conn_comps_closed_np == i).sum()
biggest_1_region_value = np.argmax(counter_uniques)

counter_uniques_without_max = np.delete(counter_uniques, biggest_region_value)
biggest_2_region_value = np.argmax(counter_uniques_without_max)