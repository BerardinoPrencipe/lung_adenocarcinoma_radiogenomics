import os
import numpy as np
import cv2
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa

import torch
import torch.nn as nn
import torch.optim as optim

from utils import imresizeNoStretch, normalize, montage, labeloverlay
from projects.ecodom.data_load import DataGen, get_balancing_weights
from networks import VNet_Xtra
from loss import dice as dice_loss

#%%
dataset_name = 'ecodom'

colors = [(0.2,0.2,0.2), (0,1,0), (1,0,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1)]
classes = ['bg', 'Paper', 'Dark-Glass', 'Light-Glass', 'Non-PET', 'Aluminum', 'PET']

abs_path = os.path.abspath('.')
dataset_folder = os.path.join(abs_path, 'data', dataset_name)

images_folder = os.path.join(dataset_folder, 'images')
labels_folder = os.path.join(dataset_folder, 'labels')

train_images = [os.path.join(images_folder, image_name) for image_name in os.listdir(images_folder) if image_name.endswith('.png') or image_name.endswith('.jpg')]
train_labels = [os.path.join(labels_folder, image_name) for image_name in os.listdir(labels_folder) if image_name.endswith('.png') or image_name.endswith('.jpg')]


DEFAULT_MODEL_DIR = os.path.join(abs_path, 'logs', dataset_name)
if not os.path.exists(DEFAULT_MODEL_DIR):
    os.makedirs(DEFAULT_MODEL_DIR)

dim_im = 512
n_channels = 3

balancing_weights = get_balancing_weights(train_labels, classes)

augmentation = iaa.SomeOf((0,2), [
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.OneOf([iaa.Affine(rotate=90),
                           iaa.Affine(rotate=180),
                           iaa.Affine(rotate=270)]),
            ])


n_val_images = 10
train_images.sort()
train_labels.sort()
train_images_ids = train_images[n_val_images:]
train_labels_ids = train_labels[n_val_images:]
valid_images_ids = train_images[:n_val_images]
valid_labels_ids = train_labels[:n_val_images]

#%% train data loader
num_samples = 50
batch_size = 1
num_workers = 0
image_size = (512,512)
num_outs = len(classes)
print('Building Training Set Loader...')
train_data_gen = DataGen(train_images_ids, train_labels_ids, augmentation=augmentation, batch_size=batch_size,
                         n_outs=num_outs, image_size=image_size)
train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=balancing_weights,
                                                               num_samples=num_samples)
train_data = torch.utils.data.DataLoader(train_data_gen, batch_size=batch_size, shuffle=False,
                                         sampler=train_sampler, num_workers=num_workers)
print('Training Loader built!')

#%% validation data loader
validation_data_gen = DataGen(train_images_ids, train_labels_ids, augmentation=None, batch_size=batch_size,
                              n_outs=num_outs, image_size=image_size)
validation_sampler = torch.utils.data.sampler.SequentialSampler(validation_data_gen)
validation_data = torch.utils.data.DataLoader(validation_data_gen, batch_size=batch_size, shuffle=False,
                                              sampler=validation_sampler, num_workers=num_workers)

#%%
dice = False
cuda = torch.cuda.is_available()
lr = 1e-2
epochs = 200
low_lr_epoch = 100
dropout = True
context = 1
# loss_weight = torch.FloatTensor([0.10, 0.90])
# if cuda: loss_weight = loss_weight.cuda()
criterion = nn.CrossEntropyLoss()

# network and optimizer
print('Building Network...')
net = VNet_Xtra(dice=dice, dropout=dropout, context=context, num_outs=num_outs)
if cuda: net = net.cuda()
optimizer = optim.Adam(net.parameters(), lr=lr)
print('Network built!')

#%%
# train loop

print('Start training...')

for epoch in range(epochs):

    epoch_start_time = time.time()
    running_loss = 0.0

    # lower learning rate
    if epoch == low_lr_epoch:
        for param_group in optimizer.param_groups:
            lr = lr / 10
            param_group['lr'] = lr

    # switch to train mode
    net.train()

    for i, data in enumerate(train_data):

        # wrap data in Variables
        inputs, labels = data
        if cuda: inputs, labels = inputs.cuda(), labels.cuda()

        # forward pass and loss calculation
        outputs = net(inputs)

        # get either dice loss or cross-entropy
        if dice:
            outputs = outputs[:, 1, :, :].unsqueeze(dim=1)
            loss = dice_loss(outputs, labels)
        else:
            labels = labels.squeeze(dim=1)
            loss = criterion(outputs, labels.long())

        # empty gradients, perform backward pass and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # save and print statistics
        running_loss += loss.data

    epoch_end_time = time.time()
    epoch_elapsed_time = epoch_end_time - epoch_start_time
    # print statistics
    if dice:
        print('  [epoch {:04d}] - train dice loss: {:.4f} - time: {:.1f}'
              .format(epoch + 1, running_loss / (i + 1), epoch_elapsed_time))
    else:
        print('  [epoch {:04d}] - train cross-entropy loss: {:.4f} - time: {:.1f}'
              .format(epoch + 1, running_loss / (i + 1), epoch_elapsed_time))

#%%
filepath_model = os.path.join(DEFAULT_MODEL_DIR, "ecodom_vnet.pht")
torch.save(net, filepath_model)

#%%
# See example
with torch.no_grad():
    image_rgb = inputs[0,:,:,:].cpu().numpy()
    image_rgb = np.transpose(image_rgb, (1,2,0))
    pred_mask = outputs[0,:,:,:].cpu().numpy()
    pred_mask = np.argmax(pred_mask, axis=0)
    image_ov_pred = labeloverlay(image_rgb, pred_mask, classes=classes, colors=colors)