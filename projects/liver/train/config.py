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
# dataset = "vessels"
# dataset = "liver"
# dataset = "segments"
dataset = "vessels_tumors"
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

epochs = 1400
augment = False
config = {
    'model_name'    : '25D',
    'augment'       : augment,
    'dropout'       : True,
    'cuda'          : cuda,
    'use_multi_gpu' : use_multi_gpu,
    'context'       : 2,                    # how many slices of context (2.5D)
    'lr'            : 1e-2,                 # learning rate
    'batch_size'    : batch_size,
    'num_samples'   : 200,                  # samples per epoch
    'low_lr_epoch'  : 100,                  # epoch where to lower learning rate
    'epochs'        : epochs,               # total number of epochs
    'val_epochs'    : 100,
    'num_outs'      : 2,
    'num_workers'   : num_workers,
    'no_softmax'    : False,                 # Set True for Softmax, False for Dice
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
    assert dataset in ["vessels", "hv", "pv", "liver", "segments"], \
        "Dataset must be in ['vessels', 'hv', 'pv', 'liver', 'segments']!"

    if dataset == "vessels":
        # Vessels
        if isLinux:
            train_folder = os.path.join(current_path_abs, 'datasets/ircadb_npy/train')
            val_folder   = os.path.join(current_path_abs, 'datasets/ircadb_npy/val')
        else:
            train_folder = 'E:/Datasets/ircadb/train'
            val_folder   = 'E:/Datasets/ircadb/val'
    elif dataset == "hv":
        train_folder = 'E:/Datasets/ircadb_hv/train'
        val_folder = 'E:/Datasets/ircadb_hv/val'
    elif dataset == "pv":
        train_folder = 'E:/Datasets/ircadb_pv/train'
        val_folder = 'E:/Datasets/ircadb_pv/val'
    elif dataset == "liver":
        # Liver LiTS
        if isLinux:
            train_folder = os.path.join(current_path_abs, 'datasets/LiTS/npy/train')
            val_folder   = os.path.join(current_path_abs, 'datasets/LiTS/npy/val')
        else:
            train_folder = 'E:/Datasets/LiTS/train'
            val_folder   = 'E:/Datasets/LiTS/val'
    elif dataset == "segments":
        if not use_masked_dataset:
            train_folder = 'E:/Datasets/LiverDecathlon/npy/train'
            val_folder   = 'E:/Datasets/LiverDecathlon/npy/val'
        else:
            train_folder = 'E:/Datasets/LiverDecathlon/npy_masked/train'
            val_folder   = 'E:/Datasets/LiverDecathlon/npy_masked/val'
    elif dataset == "vessels_tumors":
        if not use_masked_dataset:
            train_folder = 'E:/Datasets/LiverDecathlon/npy_vessels/train'
            val_folder   = 'E:/Datasets/LiverDecathlon/npy_vessels/val'
        else:
            train_folder = 'E:/Datasets/LiverDecathlon/npy_vessels_masked/train'
            val_folder   = 'E:/Datasets/LiverDecathlon/npy_vessels_masked/val'

    return train_folder, val_folder


window_hu = (-150,350)