# Try this (Windows):
# set PYTHONPATH=.
from __future__ import absolute_import

import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

current_path_abs = os.path.abspath('.')
sys.path.append(current_path_abs)
print('{} appended to sys!'.format(current_path_abs))

from utils.utils import print_dict
from projects.liver.data_util.data_load_util import *
from projects.liver.train.util import train_model, get_model_name
from projects.liver.train.config import *

def run(config, dataset):
    logs_folder = get_logs_folder(dataset)
    train_folder, val_folder = get_train_val_folders(dataset)
    criterion = get_criterion(dataset)
    config['num_outs'] = get_num_outs(dataset)

    print('Train Folder = {}\nValidation Folder = {}\nLogs Folder = {}'.format(train_folder, val_folder, logs_folder))

    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)

    print('Config')
    print_dict(config)

    # Network and optimizer
    print('Building Network...')

    if config['use_3d']:
        train_data = train_data_loader3d(train_folder, config)
        val_data_list = val_data_loader3d(val_folder, config)
        from semseg.models.vnet3d import build_VXNet3D_with_config
        net = build_VXNet3D_with_config(config)
    else:
        train_data = train_data_loader(train_folder, config)
        val_data_list = val_data_loader(val_folder, config)
        from semseg.models.vnet_v2 import build_VXNet_with_config
        net = build_VXNet_with_config(config)

    optimizer = optim.Adam(net.parameters(), lr=config['lr'])
    print('Network built!')

    # Training
    device = torch.cuda.current_device()
    net = train_model(net, optimizer, train_data, config, device=device,
                      criterion=criterion, val_data_list=val_data_list, logs_folder=logs_folder)

    # Save weights
    final_model_name = get_model_name("model_" + str(config['model_name']))
    path_final_model = os.path.join(logs_folder, final_model_name)
    torch.save(net, path_final_model)
    print('Final Model saved!')


# python projects/liver/train/train.py
# python projects/liver/train/train.py --dataset=pv
# python projects/liver/train/train.py --dataset=segments
# python projects/liver/train/train.py --dataset=vessels
# python projects/liver/train/train.py --dataset=vessels_crossval_00 &&
# python projects/liver/train/train.py --dataset=vessels_crossval_01 &&
# python projects/liver/train/train.py --dataset=vessels_crossval_02 &&
# python projects/liver/train/train.py --dataset=vessels_crossval_03
# python projects/liver/train/train.py --dataset=vessels_crossval_04
# python projects/liver/train/train.py --dataset=vessels_tumors
# python projects/liver/train/train.py --dataset=vessels_only
# python projects/liver/train/train.py --dataset=vessels_scardapane
# python projects/liver/train/train.py --dataset=vessels_scardapane_one_class

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Training on Liver Vessels")
    parser.add_argument(
        "-e",
        "--epochs",
        default=config['epochs'], type=int,
        help="Specify the number of epochs required for training"
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        default=config['num_samples'], type=int,
        help="Specify the number of samples per each epoch during training"
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=config['batch_size'], type=int,
        help="Specify the batch size"
    )
    parser.add_argument(
        "-v",
        "--val_epochs",
        default=config['val_epochs'], type=int,
        help="Specify the number of validation epochs during training"
    )
    parser.add_argument(
        "-w",
        "--num_workers",
        default=config['num_workers'], type=int,
        help="Specify the number of workers"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default=dataset,
        help="Specify the dataset on which train the net [liver | vessels | hv | pv]"
    )
    parser.add_argument(
        "--net",
        default='vnet',
        help="Specify the network to use [unet | vnet]"
    )
    parser.add_argument(
        "--lr",
        default=1e-2, type=float,
        help="Learning Rate"
    )

    args = parser.parse_args()
    dict_args = vars(args)
    for key in dict_args:
        if key == "dataset":
            dataset = dict_args[key]
            print("Dataset = {}".format(dataset))
        elif key == "net":
            net_to_use = dict_args[key]
            print("Net to use = {}".format(net_to_use))
        else:
            config[key] = dict_args[key]

    VAL_FRACTION = 6

    config['low_lr_epoch'] = config['epochs'] // VAL_FRACTION
    config['val_epochs']   = config['epochs'] // VAL_FRACTION


    run(config, dataset)