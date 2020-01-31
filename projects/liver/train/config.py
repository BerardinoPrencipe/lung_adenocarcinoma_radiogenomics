import os
import platform
import sys
import torch
import torch.nn as nn

current_path_abs = os.path.abspath('.')
sys.path.append(current_path_abs)
print('{} appended to sys!'.format(current_path_abs))

# Use local path or absolute
if 'Ubuntu' in platform.system():
    use_local_path = False
else:
    use_local_path = True

# The dataset to use!
# dataset = "vessels"
# dataset = "hv"
# dataset = "pv"
dataset = "liver"

# Hyperparams
isWindows = 'Windows' in platform.system()
if use_local_path and not isWindows:
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
    if use_local_path:
        batch_size = 4
    else:
        batch_size = 2

def get_criterion(dataset):
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

config = {
    'model_name'    : '25D',
    'augment'       : True,
    'dropout'       : True,
    'cuda'          : cuda,
    'use_multi_gpu' : use_multi_gpu,
    'context'       : 2,                    # how many slices of context (2.5D)
    'lr'            : 1e-2,                 # learning rate
    'batch_size'    : batch_size,
    'num_samples'   : 500,                  # samples per epoch
    'low_lr_epoch'  : 400,                   # epoch where to lower learning rate
    'epochs'        : 2000,                   # total number of epochs
    'val_epochs'    : 400,
    'num_outs'      : 2,
    'num_workers'   : num_workers
}
#################

def get_logs_folder(dataset):
    if use_multi_gpu:
        logs_folder = os.path.join('logs', dataset, 'multi_gpu')
    else:
        logs_folder = os.path.join('logs', dataset)
    if use_local_path:
        logs_folder = os.path.join(current_path_abs, logs_folder)
    return logs_folder

def get_train_val_folders(dataset):
    assert dataset in ["vessels", "hv", "pv", "liver"], \
        "Dataset must be in ['vessels', 'hv', 'pv', 'liver']!"

    if dataset == "vessels":
        # Vessels
        if use_local_path:
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
        if use_local_path:
            train_folder = os.path.join(current_path_abs, 'datasets/LiTS/npy/train')
            val_folder   = os.path.join(current_path_abs, 'datasets/LiTS/npy/val')
        else:
            train_folder = 'E:/Datasets/LiTS/train'
            val_folder   = 'E:/Datasets/LiTS/val'

    return train_folder, val_folder


window_hu = (-150,350)