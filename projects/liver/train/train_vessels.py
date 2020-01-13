import os
import time
import networks
import numpy as np
from subprocess import call
import platform
import time

import torch
import torch.nn as nn
import torch.optim as optim

from utils import print_dict
from projects.liver.data_util.data_load_util import train_data_loader, val_data_loader
from projects.liver.train.util import train_model
from projects.liver.train.vessels_config import logs_folder, config, train_folder, val_folder

if not os.path.exists(logs_folder):
    os.makedirs(logs_folder)

print('Config')
print_dict(config)

# network and optimizer
print('Building Network...')
net = networks.build_VNet_Xtra_with_config(config)
optimizer = optim.Adam(net.parameters(), lr=config['lr'])
print('Network built!')

# Train data loader
train_data = train_data_loader(train_folder, config)

# Validation data loader (per patient)
val_data_list = val_data_loader(val_folder, config)

# Training
net = train_model(net, optimizer, train_data, config,
                  val_data_list=val_data_list, logs_folder=logs_folder)

# save weights
final_model_name = "model_" + str(config['model_name']) + "_v1.pht"
final_model_name_state_dict = "model_state_dict_" + str(config['model_name']) + "_v1.pht"
path_final_model = os.path.join(logs_folder, final_model_name)
path_final_model_state_dict = os.path.join(logs_folder, final_model_name_state_dict)
torch.save(net, path_final_model)
torch.save(net.state_dict(), path_final_model_state_dict)

print('Final Model saved!')
