import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk

ircadb_folder = 'datasets/ircadb'
subdirs = os.listdir(ircadb_folder)
subdirs.sort()
gt_liver_paths = list()
for subdir in subdirs:
    liver_mask_path = os.path.join(ircadb_folder, subdir, 'mask', 'liver.nii')
    gt_liver_paths.append(liver_mask_path)

for idx, gt_liver_path in enumerate(gt_liver_paths):
    print('Index {} on {}'.format(idx, len(gt_liver_paths)-1))
    print('GT Liver Path  = {}'.format(gt_liver_path))
    # gt_liver_mask = nib.load(gt_liver_path)
    # spacing = gt_liver_mask.header.get_zooms()
    # spacing = np.array(spacing, dtype='double').tolist()
    # gt_liver_mask = gt_liver_mask.get_data()
    # sitk_input = sitk.GetImageFromArray(gt_liver_mask)

    sitk_input = sitk.ReadImage(gt_liver_path)
    spacing = sitk_input.GetSpacing()
    gt_liver_mask = sitk.GetArrayFromImage(sitk_input)
    vector_radius = (25, 25, 25)
    kernel = sitk.sitkBall
    gt_liver_mask_filled = sitk.BinaryMorphologicalClosing(sitk_input, vector_radius, kernel)
    print('Liver non-zero elements before filling: {}'.format(gt_liver_mask.sum()))
    gt_liver_mask = sitk.GetArrayFromImage(gt_liver_mask_filled)
    print('Liver non-zero elements after  filling: {}'.format(gt_liver_mask.sum()))
    gt_liver_mask_filled.SetSpacing(spacing)
    filename_mask_out = gt_liver_path[:-4] + "_closed.nii"
    print('Out Liver Path = {}'.format(filename_mask_out))
    sitk.WriteImage(gt_liver_mask_filled, filename_mask_out)
    print('Spacing = {}'.format(spacing))