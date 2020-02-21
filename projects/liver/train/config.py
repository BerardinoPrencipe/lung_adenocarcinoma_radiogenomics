import os
import platform
import sys
import torch
import torch.nn as nn

current_path_abs = os.path.abspath('.')
sys.path.append(current_path_abs)
print('{} appended to sys!'.format(current_path_abs))
print('Platform System = {}'.format(platform.system()))

# Use local path or absolute
if 'Ubuntu' in platform.system() or 'Linux' in platform.system():
    isLinux = True
else:
    isLinux = False

# The dataset to use!
datasets = ["liver", "vessels", "segments", "vessels_tumors"]
dataset = datasets[-1]
use_masked_dataset = False

# Hyperparams
isWindows = 'Windows' in platform.system()
if isLinux and not isWindows:
    num_workers = 4
if isWindows:
    num_workers = 0
use_multi_gpu = False
dice = True
# GPU enabled
cuda = torch.cuda.is_available()
print('CUDA is available = ', cuda)
print('Using multi-GPU   = ', use_multi_gpu)

if use_multi_gpu:
    batch_size = 10
else:
    if isLinux:
        batch_size = 8
    else:
        batch_size = 2

def get_criterion(dataset):
    return None

epochs = 1401
augment = False
config = {
    'model_name'    : '25D',
    'augment'       : augment,
    'dropout'       : True,
    'cuda'          : cuda,
    'use_multi_gpu' : use_multi_gpu,
    'context'       : 2,                    # how many slices of context (2.5D)
    'depth'         : 16,                   # 3D ONLY!
    'lr'            : 1e-2,                 # learning rate
    'batch_size'    : batch_size,
    'num_samples'   : 200,                  # samples per epoch
    'low_lr_epoch'  : 100,                  # epoch where to lower learning rate
    'epochs'        : epochs,               # total number of epochs
    'val_epochs'    : 100,
    'num_outs'      : 2,
    'num_workers'   : num_workers,
    'no_softmax'    : False,                # Set True for Softmax, False for Dice
    'image_size'    : (512, 512),           # 3D ONLY!
    'sample_xy_size': (320, 320),           # 3D ONLY!
    'use_3d'        : True,
}
#################

def get_num_outs(dataset):
    num_outs = 2
    if dataset == 'vessels_tumors':
        num_outs = 3
    elif dataset == 'segments':
        num_outs = 9
    return num_outs

def get_logs_folder(dataset):
    if use_multi_gpu:
        logs_folder = os.path.join('logs', dataset, 'multi_gpu')
    else:
        logs_folder = os.path.join('logs', dataset)
    if isLinux:
        logs_folder = os.path.join(current_path_abs, logs_folder)
    return logs_folder

def get_train_val_folders(dataset):
    assert dataset in datasets, \
        "Dataset must be in {}!".format(datasets)

    if dataset == "vessels":
        train_folder = os.path.join(current_path_abs, 'datasets/ircadb/npy/train')
        val_folder   = os.path.join(current_path_abs, 'datasets/ircadb/npy/val')
    elif dataset == "liver":
        train_folder = os.path.join(current_path_abs, 'datasets/LiTS/npy/train')
        val_folder   = os.path.join(current_path_abs, 'datasets/LiTS/npy/val')
    elif dataset == "segments":
        npy_folder = "npy" if not use_masked_dataset else "npy_masked"
        train_folder = os.path.join(current_path_abs, 'datasets/LiverDecathlon', npy_folder,'train')
        val_folder   = os.path.join(current_path_abs, 'datasets/LiverDecathlon', npy_folder,'val')
    elif dataset == "vessels_tumors":
        npy_folder = "npy_vessels" if not use_masked_dataset else "npy_vessels_masked"
        train_folder = os.path.join(current_path_abs, 'datasets/LiverDecathlon', npy_folder, 'train')
        val_folder   = os.path.join(current_path_abs, 'datasets/LiverDecathlon', npy_folder, 'val')

    return train_folder, val_folder


window_hu = (-150,350)

'''
    elif dataset == "hv":
        train_folder = 'E:/Datasets/ircadb_hv/train'
        val_folder = 'E:/Datasets/ircadb_hv/val'
    elif dataset == "pv":
        train_folder = 'E:/Datasets/ircadb_pv/train'
        val_folder = 'E:/Datasets/ircadb_pv/val'
'''

''' 
def get_criterion(dataset):
    if dataset == 'segments':
        if dice is False:
            return nn.CrossEntropyLoss().cuda()
        else:
            return None
    if dice is False:
        # cross-entropy loss: weighting of negative vs positive pixels and NLL loss layer
        loss_weight = torch.FloatTensor([0.50, 0.50])
        if dataset == "vessels":
            loss_weight = torch.FloatTensor([0.04, 0.96])
        elif dataset == "hv" or dataset == "pv":
            loss_weight = torch.FloatTensor([0.02, 0.98])
        elif dataset == "liver":
            loss_weight = torch.FloatTensor([0.10, 0.90])

        if cuda: loss_weight = loss_weight.cuda()
        criterion = nn.NLLLoss(weight=loss_weight)
        return criterion
    else:
        return None
'''