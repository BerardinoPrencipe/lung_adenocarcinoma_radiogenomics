import os
import time
import networks
import numpy as np
from subprocess import call
from loss import dice as dice_loss
from data_load import LiverDataSet
import platform
import time

import torch
import torch.nn as nn
import torch.optim as optim

from utils import use_multi_gpu_model

### variables ###
isWindows = 'Windows' in platform.system()
num_workers = 0 if isWindows else 2

model_name = '25D'

augment = True
dropout = True
use_multi_gpu = True

# using dice loss or cross-entropy loss
dice = True

# how many slices of context (2.5D)
context = 2

# learning rate, batch size, samples per epoch, epoch where to lower learning rate and total number of epochs
lr = 1e-2
if use_multi_gpu:
    batch_size = 10
else:
    batch_size = 4
num_samples = 1000
low_lr_epoch = 80
epochs = 40000
val_epochs = 800

#################

train_folder = 'E:/Datasets/LiTS/train'
val_folder = 'E:/Datasets/LiTS/val'

if use_multi_gpu:
    logs_folder = 'logs/multi_gpu'
else:
    logs_folder = 'logs'
if not os.path.exists(logs_folder):
    os.makedirs(logs_folder)

print(model_name)
print("augment="+str(augment)+" dropout="+str(dropout))
print(str(epochs) + " epochs - lr: " + str(lr) + " - batch size: " + str(batch_size))

# GPU enabled
cuda = torch.cuda.is_available()
print('CUDA is available = ', cuda)
print('Using multi-GPU   = ', use_multi_gpu)

# cross-entropy loss: weighting of negative vs positive pixels and NLL loss layer
# Tumor
# loss_weight = torch.FloatTensor([0.01, 0.99])
# Liver
loss_weight = torch.FloatTensor([0.10, 0.90])
if cuda: loss_weight = loss_weight.cuda()
criterion = nn.NLLLoss(weight=loss_weight)

# network and optimizer
print('Building Network...')
net = networks.VNet_Xtra(dice=dice, dropout=dropout, context=context)
if cuda and not use_multi_gpu: net = net.cuda()
if cuda and use_multi_gpu: net = use_multi_gpu_model(net)
optimizer = optim.Adam(net.parameters(), lr=lr)
print('Network built!')

# train data loader
print('Building Training Set Loader...')
train = LiverDataSet(directory=train_folder, augment=augment, context=context)
train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=train.getWeights(),
                                                               num_samples=num_samples)
train_data = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False,
                                         sampler=train_sampler, num_workers=num_workers)
print('Training Loader built!')

# validation data loader (per patient)
print('Building Validation Set Loader...')
val = LiverDataSet(directory=val_folder, context=context)
val_data_list = []
patients = val.getPatients()
for key in patients.keys():
    samples = patients[key]
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(samples)
    val_data = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False,
                                           sampler=val_sampler, num_workers=num_workers)
    val_data_list.append(val_data)
print('Validation Loader built!')

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
            loss = criterion(outputs, labels)

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

    # switch to eval mode
    net.eval()

    all_dice = []
    all_accuracy = []

    # only validate every 'val_epochs' epochs
    if epoch % val_epochs != 0: continue

    checkpoint_path = os.path.join(logs_folder, 'model_epoch_{:04d}.pht'.format(epoch))
    torch.save(net.state_dict(), checkpoint_path)

    # loop through patients
    eval_start_time = time.time()
    for val_data in val_data_list:

        accuracy = 0.0
        intersect = 0.0
        union = 0.0

        with torch.no_grad():
            for i, data in enumerate(val_data):

                # wrap data in Variable
                inputs, labels = data
                if cuda: inputs, labels = inputs.cuda(), labels.cuda()

                # inference
                outputs = net(inputs)

                # log softmax into softmax
                if not dice: outputs = outputs.exp()

                # round outputs to either 0 or 1
                outputs = outputs[:, 1, :, :].unsqueeze(dim=1).round()

                # accuracy
                outputs, labels = outputs.data.cpu().numpy(), labels.data.cpu().numpy()
                accuracy += (outputs == labels).sum() / float(outputs.size)

                # dice
                intersect += (outputs + labels == 2).sum()
                union += np.sum(outputs) + np.sum(labels)

        all_accuracy.append(accuracy / float(i + 1))
        all_dice.append(1 - (2 * intersect + 1e-5) / (union + 1e-5))
    eval_end_time = time.time()
    eval_elapsed_time = eval_end_time - eval_start_time
    print('    val dice loss: {:.4f} - val accuracy: {:.4f} - time: {:.1f}'
          .format(np.mean(all_dice), np.mean(all_accuracy), eval_elapsed_time))


# save weights
final_model_name = "model_" + str(model_name) + "_v2.pht"
final_model_name_state_dict = "model_state_dict_" + str(model_name) + "_v2.pht"
path_final_model = os.path.join(logs_folder, final_model_name)
path_final_model_state_dict = os.path.join(logs_folder, final_model_name_state_dict)
torch.save(net, path_final_model)
torch.save(net.state_dict(), path_final_model_state_dict)

print('Finished training...')
