import os
import SimpleITK as sitk
from utils.geometric import get_zmin_zmax, get_xmin_xmax

destination_dir = 'H:/Datasets/Liver/LiverScardapaneNew'
destination_dir_full = 'H:/Datasets/Liver/LiverScardapaneNewFull'
destination_dir = os.path.join(destination_dir, 'nii')
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)
if not os.path.exists(destination_dir_full):
    os.makedirs(destination_dir_full)

dataset_dir = 'H:/Datasets/Liver/LiverScardapane/ct_scans'
subdirs = os.listdir(dataset_dir)

#%% Reading Masks (NIFTI)
dict_patients = {}
reader_nii = sitk.ImageFileReader()
reader_nii.SetImageIO("NiftiImageIO")
for subdir in subdirs:
    print("\nPatient   : ", subdir)
    input_mask_filename = os.path.join(dataset_dir, subdir, 'mask.nii.gz')

    if not os.path.exists(input_mask_filename):
        print("No Mask")
        continue

    reader_nii.SetFileName(input_mask_filename)
    mask_sitk = reader_nii.Execute()
    mask_np = sitk.GetArrayFromImage(mask_sitk)
    z_min, z_max = get_xmin_xmax(mask_np)
    print("Z Min = {} - Max = {}".format(z_min, z_max))

    dict_patients[subdir] = (z_min, z_max)

#%% Reading Images (DCM)
writer_nii = sitk.ImageFileWriter()
writer_nii.SetImageIO("NiftiImageIO")
reader_dcm = sitk.ImageSeriesReader()
for idx, subdir in enumerate(subdirs):
    print("\n")
    print("Patient   : ", subdir)
    path_patient_dcm = os.path.join(dataset_dir, subdir)
    dicom_names = reader_dcm.GetGDCMSeriesFileNames(path_patient_dcm)
    reader_dcm.SetFileNames(dicom_names)
    image_sitk = reader_dcm.Execute()
    size = image_sitk.GetSize()
    origin = image_sitk.GetOrigin()
    direction = image_sitk.GetDirection()

    spacing = image_sitk.GetSpacing()
    print("Image size: ", size)
    print("Spacing   : ", spacing)
    print("Origin    : ", origin)
    print("Direction : ", direction)

    image_np = sitk.GetArrayFromImage(image_sitk)
    print("Max Value : ", image_np.max())
    print("Min Value : ", image_np.min())

    # Image
    try:
        zmin, zmax = dict_patients[subdir]
        image_np_filtered = image_np[zmin:zmax]
        image_sitk_filtered = sitk.GetImageFromArray(image_np_filtered)
        image_sitk_filtered.SetDirection(direction)
        image_sitk_filtered.SetOrigin(origin)
        image_sitk_filtered.SetSpacing(spacing)

        image_sitk_full = sitk.GetImageFromArray(image_np)
        image_sitk_full.SetDirection(direction)
        image_sitk_full.SetOrigin(origin)
        image_sitk_full.SetSpacing(spacing)

        destination_filename = os.path.join(destination_dir, "ct_scan_{:02d}.nii.gz".format(idx))
        destination_filename_full = os.path.join(destination_dir_full, "ct_scan_{:02d}.nii.gz".format(idx))

        print("Writing to ", destination_filename)
        writer_nii.SetFileName(destination_filename)
        writer_nii.Execute(image_sitk_filtered)

        print("Writing to ", destination_filename_full)
        writer_nii.SetFileName(destination_filename_full)
        writer_nii.Execute(image_sitk_full)


    except KeyError as e:
        print("[Image] Missing key: ", subdir)

    # Mask
    try:
        zmin, zmax = dict_patients[subdir]
        input_mask_filename = os.path.join(dataset_dir, subdir, 'mask.nii.gz')
        reader_nii.SetFileName(input_mask_filename)
        mask_sitk = reader_nii.Execute()
        mask_np = sitk.GetArrayFromImage(mask_sitk)

        mask_np_filtered = mask_np[zmin:zmax]
        mask_sitk_filtered = sitk.GetImageFromArray(mask_np_filtered)
        mask_sitk_filtered.SetDirection(direction)
        mask_sitk_filtered.SetOrigin(origin)
        mask_sitk_filtered.SetSpacing(spacing)

        mask_sitk_full = sitk.GetImageFromArray(mask_np)
        mask_sitk_full.SetDirection(direction)
        mask_sitk_full.SetOrigin(origin)
        mask_sitk_full.SetSpacing(spacing)

        destination_filename = os.path.join(destination_dir, "ct_mask_{:02d}.nii.gz".format(idx))
        destination_filename_full = os.path.join(destination_dir_full, "ct_mask_{:02d}.nii.gz".format(idx))

        print("Writing to ", destination_filename)
        writer_nii.SetFileName(destination_filename)
        writer_nii.Execute(mask_sitk_filtered)

        print("Writing to ", destination_filename_full)
        writer_nii.SetFileName(destination_filename_full)
        writer_nii.Execute(mask_sitk_full)



    except KeyError as e:
        print("[Mask ] Missing key: ", subdir)
