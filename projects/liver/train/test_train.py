import os
import sys
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

current_path_abs = os.path.abspath('.')
sys.path.append(current_path_abs)
print('{} appended to sys!'.format(current_path_abs))

from utils import print_dict, montage, labeloverlay
from projects.liver.data_util.data_load_util import *
from projects.liver.train.util import train_model, get_model_name
from projects.liver.train.config import *


augmentation = iaa.Sequential( [
                iaa.GaussianBlur(sigma=(0.0, 0.03)),
                iaa.ElasticTransformation(alpha=(1,2), sigma=2.5),
                iaa.Multiply((0.99, 1.01)),
                iaa.Rotate(rotate=(-15,15))
            ])
config['augmentation'] = augmentation

dataset = 'segments'
logs_folder = get_logs_folder(dataset)
train_folder, val_folder = get_train_val_folders(dataset)
criterion = get_criterion(dataset)
config['num_outs'] = get_num_outs(dataset)

print('Train Folder = {}\nValidation Folder = {}\nLogs Folder = {}'.format(train_folder, val_folder, logs_folder))

if not os.path.exists(logs_folder):
    os.makedirs(logs_folder)

print('Config')
print_dict(config)

if config['use_3d']:
    train_data = train_data_loader3d(train_folder, config)
    val_data_list = val_data_loader3d(val_folder, config)
else:
    train_data = train_data_loader(train_folder, config)
    val_data_list = val_data_loader(val_folder, config)

for i, data in enumerate(train_data):
    inputs, labels = data

    print('Inputs shape = {}'.format(inputs.shape))
    print('Labels shape = {}'.format(labels.shape))

    inputs, labels = inputs.cpu().numpy(), labels.cpu().numpy()

    print('Min = {}'.format(inputs.min()))
    print('Max = {}'.format(inputs.max()))


    img = inputs[0,0,:,:]
    lab = labels[0, 0, :, :]
    ov  = inputs[0,0,:,:] + labels[0, 0, :, :]
    imgs = [img, lab, ov]
    imgs_mont = montage(imgs, dim=(1,3))

    plt.figure()
    plt.imshow(imgs_mont, cmap='gray')
    plt.show()

    if i == 5:
        break