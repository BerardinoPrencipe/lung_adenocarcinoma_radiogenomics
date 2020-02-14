import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

# net_to_use = 'unet'
net_to_use = 'vnet'

current_path_abs = os.path.abspath('.')
sys.path.append(current_path_abs)
print('{} appended to sys!'.format(current_path_abs))

from utils import print_dict
from projects.liver.data_util.data_load_util import train_data_loader, val_data_loader
from projects.liver.train.util import train_model, get_model_name
from projects.liver.train.config import config, dataset, \
                                        get_logs_folder, get_train_val_folders, get_criterion

def run(config, dataset):
    logs_folder = get_logs_folder(dataset)
    train_folder, val_folder = get_train_val_folders(dataset)
    criterion = get_criterion(dataset)

    print('Train Folder = {}\nValidation Folder = {}\nLogs Folder = {}'.format(train_folder, val_folder, logs_folder))

    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)

    print('Config')
    print_dict(config)

    # Network and optimizer
    print('Building Network...')

    if net_to_use == 'vnet':
        # net = semseg.models.vnet.build_VNet_Xtra_with_config(config, criterion)
        from semseg.models.vnet_v2 import build_VXNet_with_config
        net = build_VXNet_with_config(config)
    elif net_to_use == 'unet':
        from semseg.models.unet import build_UNet_with_config
        net = build_UNet_with_config(config)
    else:
        net = None


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
# python projects/liver/train/train.py --dataset=segments

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
        default=net_to_use,
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

    config['low_lr_epoch'] = config['epochs'] // 5
    config['val_epochs']   = config['epochs'] // 5

    if dataset == "segments":
        config['num_outs'] = 9

    run(config, dataset)