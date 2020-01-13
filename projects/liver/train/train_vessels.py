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
from projects.liver.train.util import train_model, get_model_name
from projects.liver.train.vessels_config import logs_folder, config, train_folder, val_folder

def run(config):
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
    final_model_name = get_model_name("model_" + str(config['model_name']))
    path_final_model = os.path.join(logs_folder, final_model_name)
    torch.save(net, path_final_model)

    final_model_name_state_dict = get_model_name("model_state_dict_" + str(config['model_name']))
    path_final_model_state_dict = os.path.join(logs_folder, final_model_name_state_dict)
    torch.save(net.state_dict(), path_final_model_state_dict)

    print('Final Model saved!')


# python projects/liver/train/train_vessels.py
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Training on Liver Vessels")
    parser.add_argument(
        "-e",
        "--epochs",
        default=config['epochs'],
        help="Specify the number of epochs required for training"
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        default=config['num_samples'],
        help="Specify the number of samples per each epoch during training"
    )
    parser.add_argument(
        "-v",
        "--val_epochs",
        default=config['val_epochs'],
        help="Specify the number of validation epochs during training"
    )

    args = parser.parse_args()
    dict_args = vars(args)
    for key in dict_args:
        config[key] = dict_args[key]

    run(config)