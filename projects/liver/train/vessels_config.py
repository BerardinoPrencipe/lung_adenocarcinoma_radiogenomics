import platform
import torch
import torch.nn as nn

### variables ###
isWindows = 'Windows' in platform.system()
num_workers = 0 if isWindows else 2
use_multi_gpu = False
# GPU enabled
cuda = torch.cuda.is_available()
print('CUDA is available = ', cuda)
print('Using multi-GPU   = ', use_multi_gpu)

if use_multi_gpu:
    batch_size = 10
else:
    batch_size = 4

# cross-entropy loss: weighting of negative vs positive pixels and NLL loss layer
# Vessels
loss_weight = torch.FloatTensor([0.04, 0.96])
if cuda: loss_weight = loss_weight.cuda()
criterion = nn.NLLLoss(weight=loss_weight)

config = {
    'model_name'    : '25D',
    'augment'       : True,
    'dropout'       : True,
    'cuda'          : cuda,
    'use_multi_gpu' : use_multi_gpu,
    'dice'          : True,                 # using dice loss or cross-entropy loss
    'context'       : 2,                    # how many slices of context (2.5D)
    'lr'            : 1e-2,                 # learning rate
    'batch_size'    : batch_size,
    'num_samples'   : 500,                  # samples per epoch
    'low_lr_epoch'  : 200,                   # epoch where to lower learning rate
    'epochs'        : 1000,                   # total number of epochs
    'val_epochs'    : 200,
    'num_outs'      : 2,
    'criterion'     : criterion,
    'num_workers'   : num_workers
}
#################

train_folder = 'E:/Datasets/ircadb/train'
val_folder = 'E:/Datasets/ircadb/val'

if use_multi_gpu:
    logs_folder = 'logs/vessels/multi_gpu'
else:
    logs_folder = 'logs/vessels'