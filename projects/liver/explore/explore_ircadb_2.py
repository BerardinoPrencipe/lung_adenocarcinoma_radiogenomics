import os
import nibabel as nib
import numpy as np

path_ircad_nifti_dataset = 'H:\\Datasets\\Liver\\3Dircadb1_nifti'
subdirs = os.listdir(path_ircad_nifti_dataset)
subdirs = [subdir for subdir in subdirs if not subdir.endswith('json')]
print("Subdirs = ", subdirs)

VESSELS_CLASS = (2, 3)

fg_counter = 0
bg_counter = 1

for idx, subdir in enumerate(subdirs):
    print("Iter {} on {}".format(idx+1, len(subdirs)))
    mask_filename = os.path.join(path_ircad_nifti_dataset, subdir, 'mask', 'mask.nii')

    mask_data = nib.load(mask_filename)

    mask_data = mask_data.get_data()
    mask_data_hv = (mask_data == VESSELS_CLASS[0]).astype(np.uint8)
    mask_data_pv = (mask_data == VESSELS_CLASS[1]).astype(np.uint8)
    mask_data = np.logical_or(mask_data_hv, mask_data_pv)

    fg_counter += (mask_data == 1).sum()
    bg_counter += (mask_data == 0).sum()

    print("BG cnt = ", bg_counter)
    print("FG cnt = ", fg_counter)
    bg_fg_ratio = bg_counter / fg_counter
    print("BG/FG  = ", bg_fg_ratio)
