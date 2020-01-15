import os
import zipfile
import SimpleITK as sitk
import numpy as np
from utils import normalize_data

def pre_process_dicom(image_data, window_hu):
    image_data = normalize_data(image_data, window_hu[0], window_hu[1])
    image_data_ = (image_data * 255).astype(np.uint8)
    return image_data_

do_unzip = False

path_ircadb     = 'F:/Datasets/3Dircadb1'
path_ircadb_out = 'datasets/ircadb'
subfolders = os.listdir(path_ircadb)

#%% Unzip
if do_unzip:
    for subfolder in subfolders:
        print('\n\nSubfolder = ', subfolder)
        files = os.listdir(os.path.join(path_ircadb, subfolder))
        zip_files = [file for file in files if file.endswith(".zip")]
        for zip_file in zip_files:
            subfolder_path = os.path.join(os.path.join(path_ircadb, subfolder))
            print('Extracting ', zip_file, ' in ', subfolder_path)
            with zipfile.ZipFile(os.path.join(subfolder_path, zip_file)) as zip_ref:
                zip_ref.extractall(subfolder_path)


#%% Create list of dicts
patients = list()
venacava_cnt     = 0
venoussystem_cnt = 0
portalvein_cnt   = 0
artery_cnt       = 0
vessels_cnt      = 0
for subfolder in subfolders:
    files = os.listdir(os.path.join(path_ircadb, subfolder))
    subdirs = [os.path.join(path_ircadb, subfolder, file) for file in files
               if os.path.isdir(os.path.join(path_ircadb, subfolder, file))]
    patient_dicom = [subdir for subdir in subdirs if 'PATIENT_DICOM' in subdir][0]
    masks_dicom = [subdir for subdir in subdirs if 'MASKS_DICOM' in subdir][0]
    subdir_masks_dicom = os.listdir(masks_dicom)

    liver_subdir    = []
    vessels_subdirs = list()
    vessels_cnt_has_incremented = False
    for file in subdir_masks_dicom:
        if file=='liver':
            liver_subdir = file
        elif file=='venacava' or file=='venoussystem' or file=='portalvein' or file=='artery':
            if not vessels_cnt_has_incremented:
                vessels_cnt += 1
                vessels_cnt_has_incremented = True
            if file=='venacava':
                vessels_subdirs.append(file)
                venacava_cnt += 1
            elif file=='venoussystem':
                vessels_subdirs.append(file)
                venoussystem_cnt += 1
            elif file=='portalvein':
                vessels_subdirs.append(file)
                portalvein_cnt += 1
            elif file=='artery':
                vessels_subdirs.append(file)
                artery_cnt += 1
    patient = {
        'images_folder'  : patient_dicom,
        'masks_folder'   : masks_dicom,
        'liver_subdir'   : liver_subdir,
        'vessels_subdirs': vessels_subdirs,
    }
    patients.append(patient)

print('Vessels       counter = {:2d}'.format(vessels_cnt))
print('Vena Cava     counter = {:2d}'.format(venacava_cnt))
print('Venous System counter = {:2d}'.format(venoussystem_cnt))
print('Portal Vein   counter = {:2d}'.format(portalvein_cnt))
print('Artery        counter = {:2d}'.format(artery_cnt))

#%% Read DICOM
# Legenda
# 1 --> Liver           --- 'liver'
# 2 --> Vena Cava       --- 'venacava'
# 2 --> Venous System   --- 'venoussystem'
# 3 --> Portal Vein     --- 'portalvein'
# 4 --> Artery          --- 'artery'

conversion_dict = {
    'liver'         : 1,
    'venacava'      : 2,
    'venoussystem'  : 2,
    'portalvein'    : 3,
    'artery'        : 4,
}


def get_dir_num(patient):
    return int(os.path.basename(os.path.dirname(patient['images_folder']))[10:])


for patient in patients:

    parent_dir_num = get_dir_num(patient)
    print('Parent dir num = ', parent_dir_num)

    dicom_dir_images = patient['images_folder']
    dicom_dir_liver = os.path.join(patient['masks_folder'], patient['liver_subdir'])
    vessels_subdirs = patient['vessels_subdirs']
    dicom_dir_vessels = [os.path.join(patient['masks_folder'], vessel_subdir) for vessel_subdir in vessels_subdirs]
    intensity_values = [conversion_dict[vessels_subdir] for vessels_subdir in vessels_subdirs]

    print('Reading DICOM directory: ', dicom_dir_images)
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir_images)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    size = image.GetSize()
    print("Image size: ", size[0], size[1], size[2])
    spacing = image.GetSpacing()
    image_data = sitk.GetArrayFromImage(image)

    print('Reading DICOM directory: ', dicom_dir_liver)
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir_liver)
    reader.SetFileNames(dicom_names)
    liver = reader.Execute()
    liver_data = sitk.GetArrayFromImage(liver)
    liver_data = liver_data / np.max(liver_data)

    vessels_global_mask = np.zeros(image_data.shape)
    for dicom_dir_vessel, intensity_value in zip(dicom_dir_vessels, intensity_values):
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir_vessel)
        reader.SetFileNames(dicom_names)
        vessels_mask = reader.Execute()
        vessels_data = sitk.GetArrayFromImage(vessels_mask)
        vessels_data = vessels_data / np.max(vessels_data) * intensity_value
        vessels_global_mask[vessels_global_mask == 0] += vessels_data[vessels_global_mask == 0]
    vessels_global_mask = vessels_global_mask.astype(np.uint8)
    print('Unique Values Vessels = ', np.unique(vessels_global_mask))

    liver_vessels_mask = liver_data
    liver_vessels_mask[vessels_global_mask != 0] = vessels_global_mask[vessels_global_mask != 0]
    liver_vessels_mask = liver_vessels_mask.astype(np.uint8)
    print('Unique Values Mask    = ', np.unique(liver_vessels_mask))

    itk_mask_out = sitk.GetImageFromArray(liver_vessels_mask)

    itk_liver_mask_out = sitk.GetImageFromArray(1*(liver_vessels_mask==conversion_dict['liver']).astype(np.uint8))
    itk_hv_mask_out = sitk.GetImageFromArray(1*(liver_vessels_mask==conversion_dict['venacava']).astype(np.uint8))
    itk_pv_mask_out = sitk.GetImageFromArray(1*(liver_vessels_mask==conversion_dict['portalvein']).astype(np.uint8))
    itk_artery_mask_out = sitk.GetImageFromArray(1*(liver_vessels_mask==conversion_dict['artery']).astype(np.uint8))

    path_out = os.path.join(path_ircadb_out, 'patient-{:02d}'.format(parent_dir_num))
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    path_mask  = os.path.join(path_out, 'mask')
    if not os.path.exists(path_mask):
        os.makedirs(path_mask)
    path_image = os.path.join(path_out, 'image')
    if not os.path.exists(path_image):
        os.makedirs(path_image)
    filename_mask_out = os.path.join(path_mask, 'mask.nii')
    itk_mask_out.SetSpacing(spacing)
    sitk.WriteImage(itk_mask_out, filename_mask_out)

    filename_liver_out = os.path.join(path_mask, 'liver.nii')
    itk_liver_mask_out.SetSpacing(spacing)
    sitk.WriteImage(itk_liver_mask_out, filename_liver_out)

    filename_hv_out = os.path.join(path_mask, 'hv.nii')
    itk_hv_mask_out.SetSpacing(spacing)
    sitk.WriteImage(itk_hv_mask_out, filename_hv_out)

    filename_pv_out = os.path.join(path_mask, 'pv.nii')
    itk_pv_mask_out.SetSpacing(spacing)
    sitk.WriteImage(itk_pv_mask_out, filename_pv_out)

    filename_artery_out = os.path.join(path_mask, 'artery.nii')
    itk_artery_mask_out.SetSpacing(spacing)
    sitk.WriteImage(itk_artery_mask_out, filename_artery_out)

    filename_image_out = os.path.join(path_image, 'image.nii')

    # image_data_ = pre_process_dicom(image_data)
    image_data_ = image_data.astype(np.int16)
    image_new = sitk.GetImageFromArray(image_data_)
    image_new.SetSpacing(spacing)
    sitk.WriteImage(image_new, filename_image_out)

show_mask = False
if show_mask:
    if ( not "SITK_NOSHOW" in os.environ ):
        sitk.Show(itk_mask_out, "Dicom Series")

cnt_2 = 0
cnt_3 = 0
for patient in patients:
    num_vessels_type = len(patient['vessels_subdirs'])
    num = get_dir_num(patient)
    print("Patient {:02d}/{:02d} - Num Vessels Types: {}".format(num, len(patients), num_vessels_type))
    print("Vessels subdirs: ", patient['vessels_subdirs'])
    if num_vessels_type == 2:
        cnt_2 += 1
    if num_vessels_type == 3:
        cnt_3 += 1
print("Counter 1 = ", len(patients) - cnt_2 - cnt_3)
print("Counter 2 = ", cnt_2)
print("Counter 3 = ", cnt_3)