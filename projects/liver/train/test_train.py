import os
import sys
import matplotlib.pyplot as plt

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

current_path_abs = os.path.abspath('.')
sys.path.append(current_path_abs)
print('{} appended to sys!'.format(current_path_abs))

from util.utils import print_dict, montage, labeloverlay
from projects.liver.data_util.data_load_util import *
from projects.liver.train.util import train_model, get_model_name
from projects.liver.train.config import *

from albumentations import (
    GaussianBlur, ElasticTransform, MultiplicativeNoise, Rotate,
    Compose
)

augmentation = Compose([
        # GaussianBlur(p=1.),
        # ElasticTransform(alpha=2, sigma=3, alpha_affine=0, p=1.),
        # MultiplicativeNoise(multiplier=(0.98,1.02), per_channel=False, elementwise=True),
        Rotate(limit=(-5,5),border_mode=cv2.BORDER_CONSTANT,value=-200,mask_value=0,p=1.),
    ]
)
augmentation = None

config['augmentation'] = augmentation

dataset = 'vessels_no_norm'
dataset = 'liver'
dataset = 'liver_no_norm'
logs_folder = get_logs_folder(dataset)
train_folder, val_folder = get_train_val_folders(dataset)
criterion = get_criterion(dataset)
config['num_outs'] = get_num_outs(dataset)
config['do_normalize'] = True

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

    print(f'Min = {inputs.min()} - Max = {inputs.max()}')

    img = inputs[0,0,:,:]
    lab = labels[0,0,:,:]
    ov  = inputs[0,0,:,:] + labels[0, 0, :, :]
    imgs = [img, lab, ov]
    imgs_mont = montage(imgs, dim=(1,3))

    plt.figure()
    plt.imshow(imgs_mont, cmap='gray')
    plt.show()

    if i == 5:
        break