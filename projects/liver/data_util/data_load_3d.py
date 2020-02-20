import os
import torch
import torch.utils.data as data_utils
from projects.liver.data_util.data_load import LiverDataSet

class DataLoader3D(LiverDataSet):

    def __init__(self, directory, augment=False, context=0):
        assert(context%16 == 0, 'Context {} is not divisible per 16!'.format(context))
        super().__init__(directory, augment=augment, context=context)

    def __getitem__(self, idx):
        return load_patch_3d(self.data_files, idx, self.context, self.directory, self.augment)

#TODO: handle augmentation
def load_patch_3d(data_files, idx, context, directory, augment):

    max_idx = get_max_index(data_files[idx][0])
    if idx+context>=max_idx:
        slices = range(idx,idx+context)
    else:
        slices = range(max_idx-context+1,max_idx)

file_names = ['patient_2-10', 'patient_2-11', 'patient_2-12']
def get_max_index(file_names):
    file_idxs = [file_name.split('_')[1].split('-')[1] for file_name in file_names]
    return max(file_idxs)