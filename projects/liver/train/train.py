import os
import time
import semseg.models.vnet
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
from projects.liver.train.config import config, dataset, \
                                                get_logs_folder, get_train_val_folders, get_criterion

def run(config, dataset):
    logs_folder = get_logs_folder(dataset)
    train_folder, val_folder = get_train_val_folders(dataset)
    criterion = get_criterion(dataset)

    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)

    print('Config')
    print_dict(config)

    # Network and optimizer
    print('Building Network...')
    net = semseg.models.vnet.build_VNet_Xtra_with_config(config, criterion)
    optimizer = optim.Adam(net.parameters(), lr=config['lr'])
    print('Network built!')

    # Train data loader
    train_data = train_data_loader(train_folder, config)

    # Validation data loader (per patient)
    val_data_list = val_data_loader(val_folder, config)

    # Training
    net = train_model(net, optimizer, train_data, config,
                      criterion=criterion, val_data_list=val_data_list, logs_folder=logs_folder)

    # Save weights
    final_model_name = get_model_name("model_" + str(config['model_name']))
    path_final_model = os.path.join(logs_folder, final_model_name)
    torch.save(net, path_final_model)
    print('Final Model saved!')


# python projects/liver/train/train.py
# python projects/liver/train/train.py --dataset=pv

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
    parser.add_argument(
        "-d",
        "--dataset",
        default=dataset,
        help="Specify the dataset on which train the net [liver | vessels | hv | pv]"
    )

    args = parser.parse_args()
    dict_args = vars(args)
    for key in dict_args:
        if key == "dataset":
            dataset = dict_args[key]
            print("Dataset = {}".format(dataset))
        else:
            config[key] = dict_args[key]

    run(config, dataset)