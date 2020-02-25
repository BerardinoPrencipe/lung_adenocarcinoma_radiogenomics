import os
import torch
import numpy as np
from projects.liver.data_util.data_load import LiverDataSet

class DataLoader3D(LiverDataSet):

    def __init__(self, directory, augment=None, context=0, image_size=(384,384)):
        assert context % 16 == 0, 'Context {} is not divisible per 16!'.format(context)
        super().__init__(directory, augment=augment, context=context)
        self.image_size = image_size
    def __getitem__(self, idx):
        return load_patch_3d(self.data_files, idx, self.context, self.directory, self.image_size, self.augment)

def load_patch_3d(data_files, idx, context, directory, image_size, augment=None):

    max_idx = get_max_index(data_files,idx)
    if idx+context < max_idx:
        slices = range(idx, idx+context)
    else:
        slices = range(max_idx-context+2, max_idx+2)
    inputs = []
    labels = []
    for slice in slices:
        input, label = np.load(os.path.join(directory,data_files[slice][0])),np.load(os.path.join(directory,data_files[slice][1]))
        input, label = np.expand_dims(input, 0), np.expand_dims(label, 0)
        inputs.append(input)
        labels.append(label)

    inputs = np.concatenate(inputs,0)
    labels = np.concatenate(labels,0)

    # TODO: handle augmentation
    if augment is not None:
        pass

    inputs, labels = np.expand_dims(inputs, 0), np.expand_dims(labels, 0)

    cx, cy = image_size
    _, _, dx, dy = inputs.shape
    if dx - cx != 0 and dy - cy != 0:
        rx, ry = np.random.randint(0, dx - cx), np.random.randint(0, dy - cy)
        inputs = inputs[:, :, rx:rx + cx, ry:ry + cy]
        labels = labels[:, :, rx:rx + cx, ry:ry + cy]

    inputs, labels = torch.from_numpy(inputs).float(), torch.from_numpy(labels.astype(np.uint8)).long()
    return (inputs,labels)

def get_max_index(file_names, patient):
    patient = file_names[patient][0].split('-')[1].split('_')[0]
    file_names = [file_name for file_name in file_names if file_name[0].split('-')[1].split('_')[0] == patient]
    file_idxs = [int(file_name[0].split('-')[1].split('_')[1][:-4]) for file_name in file_names]
    return max(file_idxs)
