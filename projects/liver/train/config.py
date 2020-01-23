import os
import platform
import torch
import torch.nn as nn

# The dataset to use!
# dataset = "vessels"
# dataset = "hv"
# dataset = "pv"
dataset = "liver"

# Hyperparams
isWindows = 'Windows' in platform.system()
num_workers = 0 if isWindows else 2
use_multi_gpu = False
dice = True
# GPU enabled
cuda = torch.cuda.is_available()
print('CUDA is available = ', cuda)
print('Using multi-GPU   = ', use_multi_gpu)

if use_multi_gpu:
    batch_size = 10
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
    'low_lr_epoch'  : 200,                   # epoch where to lower learning rate
    'epochs'        : 1000,                   # total number of epochs
    'val_epochs'    : 200,
    'num_outs'      : 2,
    'num_workers'   : num_workers
}
#################

def get_logs_folder(dataset):
    if use_multi_gpu:
        logs_folder = os.path.join('logs', dataset, 'multi_gpu')
    else:
        logs_folder = os.path.join('logs', dataset)
    return logs_folder

def get_train_val_folders(dataset):
    assert dataset in ["vessels", "hv", "pv", "liver"], \
        "Dataset must be in ['vessels', 'hv', 'pv', 'liver']!"

    if dataset == "vessels":
        # Vessels
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
        train_folder = 'E:/Datasets/LiTS/train'
        val_folder   = 'E:/Datasets/LiTS/val'

    return train_folder, val_folder


window_hu = (-150,350)