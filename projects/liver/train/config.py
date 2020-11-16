import os
import platform
import sys
import torch
import torch.nn as nn
import cv2
from albumentations import (
    GaussianBlur, ElasticTransform, MultiplicativeNoise, Rotate,
    Compose, CLAHE, RandomBrightnessContrast
)

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
datasets = ["liver", "liver_no_norm",
            "vessels", "vessels_no_norm",
            "vessels_crossval_00", "vessels_crossval_01", "vessels_crossval_02", "vessels_crossval_03", "vessels_crossval_04",
            "spleen_crossval_00", "spleen_crossval_01", "spleen_crossval_02", "spleen_crossval_03",
            "segments",
            "vessels_tumors", "vessels_only",
            "vessels_scardapane", "vessels_scardapane_one_class",
            "covid19"]
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
        # batch_size = 2
        batch_size = 4

def get_criterion(dataset):
    return None

# epochs  = 2001
epochs  = 1001
use_3d  = False
p = 0.1
do_normalize = False

augmentation = Compose([
        GaussianBlur(p=p),
        ElasticTransform(alpha=2, sigma=3, alpha_affine=0, p=p),
        MultiplicativeNoise(multiplier=(0.97,1.03), per_channel=False, elementwise=True, p=p),
        Rotate(limit=(-10,10),border_mode=cv2.BORDER_CONSTANT,value=0,mask_value=0, p=p),
        RandomBrightnessContrast(contrast_limit=0.15, brightness_limit=0.15, p=p)
    ]
)

config = {
    'model_name'    : '25D' if not use_3d else '3D',
    'augmentation'  : augmentation,
    'do_normalize'  : do_normalize,
    'dropout'       : False,
    'cuda'          : cuda,
    'use_multi_gpu' : use_multi_gpu,
    'context'       : 2,                      # how many slices of context (2.5D)
    'depth'         : 16 if use_3d else None, # 3D ONLY!
    'lr'            : 1e-2,                   # learning rate
    'batch_size'    : batch_size,
    'num_samples'   : 200,                    # samples per epoch
    'low_lr_epoch'  : 100,                    # epoch where to lower learning rate
    'epochs'        : epochs,                 # total number of epochs
    'val_epochs'    : 100,
    'num_outs'      : 2,
    'num_workers'   : num_workers,
    'no_softmax'    : False,                # Set True for Softmax, False for Dice
    'image_size'    : (512, 512),           # 3D ONLY!
    'sample_xy_size': (320, 320),           # 3D ONLY!
    'use_3d'        : use_3d,
}
#################

def get_num_outs(dataset):
    num_outs = 2
    if dataset == 'vessels_tumors':
        num_outs = 3 # Background, Vessels, Tumors
    elif dataset == 'vessels_scardapane':
        num_outs = 3 # Background, HV, PV
    elif dataset == 'covid19':
        num_outs = 3 # Background, Ground Glass, Consolidation
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
    elif dataset == "vessels_crossval_00":
        train_folder = os.path.join(current_path_abs, 'datasets/ircadb/npy_crossval_00/train')
        val_folder   = os.path.join(current_path_abs, 'datasets/ircadb/npy_crossval_00/val')
    elif dataset == "vessels_crossval_01":
        train_folder = os.path.join(current_path_abs, 'datasets/ircadb/npy_crossval_01/train')
        val_folder   = os.path.join(current_path_abs, 'datasets/ircadb/npy_crossval_01/val')
    elif dataset == "vessels_crossval_02":
        train_folder = os.path.join(current_path_abs, 'datasets/ircadb/npy_crossval_02/train')
        val_folder   = os.path.join(current_path_abs, 'datasets/ircadb/npy_crossval_02/val')
    elif dataset == "vessels_crossval_03":
        train_folder = os.path.join(current_path_abs, 'datasets/ircadb/npy_crossval_03/train')
        val_folder   = os.path.join(current_path_abs, 'datasets/ircadb/npy_crossval_03/val')
    elif dataset == "vessels_crossval_04":
        train_folder = os.path.join(current_path_abs, 'datasets/ircadb/npy_crossval_04/train')
        val_folder = os.path.join(current_path_abs, 'datasets/ircadb/npy_crossval_04/val')
    elif dataset == "spleen_crossval_00":
        train_folder = os.path.join(current_path_abs, 'datasets/Task09_Spleen/npy_crossval_000/train')
        val_folder   = os.path.join(current_path_abs, 'datasets/Task09_Spleen/npy_crossval_000/val')
    elif dataset == "spleen_crossval_01":
        train_folder = os.path.join(current_path_abs, 'datasets/Task09_Spleen/npy_crossval_001/train')
        val_folder   = os.path.join(current_path_abs, 'datasets/Task09_Spleen/npy_crossval_001/val')
    elif dataset == "spleen_crossval_02":
        train_folder = os.path.join(current_path_abs, 'datasets/Task09_Spleen/npy_crossval_002/train')
        val_folder   = os.path.join(current_path_abs, 'datasets/Task09_Spleen/npy_crossval_002/val')
    elif dataset == "spleen_crossval_03":
        train_folder = os.path.join(current_path_abs, 'datasets/Task09_Spleen/npy_crossval_003/train')
        val_folder   = os.path.join(current_path_abs, 'datasets/Task09_Spleen/npy_crossval_003/val')
    elif dataset == "vessels_no_norm":
        train_folder = os.path.join(current_path_abs, 'datasets/ircadb/npy_no_norm/train')
        val_folder = os.path.join(current_path_abs, 'datasets/ircadb/npy_no_norm/val')
    elif dataset == "liver":
        train_folder = os.path.join(current_path_abs, 'datasets/LiTS/npy/train')
        val_folder   = os.path.join(current_path_abs, 'datasets/LiTS/npy/val')
    elif dataset == "liver_no_norm":
        train_folder = os.path.join(current_path_abs, 'datasets/LiTS/npy_no_norm/train')
        val_folder = os.path.join(current_path_abs, 'datasets/LiTS/npy_no_norm/val')
    elif dataset == "segments":
        npy_folder = "npy" if not use_masked_dataset else "npy_masked"
        train_folder = os.path.join(current_path_abs, 'datasets/LiverDecathlon', npy_folder,'train')
        val_folder   = os.path.join(current_path_abs, 'datasets/LiverDecathlon', npy_folder,'val')
    elif dataset == "vessels_tumors":
        npy_folder = "npy_vessels" if not use_masked_dataset else "npy_vessels_masked"
        train_folder = os.path.join(current_path_abs, 'datasets/LiverDecathlon', npy_folder, 'train')
        val_folder   = os.path.join(current_path_abs, 'datasets/LiverDecathlon', npy_folder, 'val')
    elif dataset == "vessels_only":
        train_folder = os.path.join(current_path_abs, 'datasets/LiverDecathlon', 'npy_vessels_only','train')
        val_folder = os.path.join(current_path_abs, 'datasets/LiverDecathlon', 'npy_vessels_only','val')
    elif dataset == "vessels_scardapane":
        dataset_path = 'H:/Datasets/Liver/LiverScardapaneNew/npy'
        train_folder = os.path.join(dataset_path, 'train')
        val_folder = os.path.join(dataset_path, 'val')
    elif dataset == "vessels_scardapane_one_class":
        dataset_path = 'H:/Datasets/Liver/LiverScardapaneNew/npy_one_class'
        train_folder = os.path.join(dataset_path, 'train')
        val_folder = os.path.join(dataset_path, 'val')
    elif dataset == "covid19":
        train_folder = os.path.join(current_path_abs, 'datasets/CoVID19/npy/train')
        val_folder = os.path.join(current_path_abs, 'datasets/CoVID19/npy/val')

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