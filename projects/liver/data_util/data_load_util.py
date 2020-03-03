import torch
from projects.liver.data_util.data_load import LiverDataSet
from projects.liver.data_util.data_load_3d import DataLoader3D


def train_data_loader(train_folder, config):
    print('Building Training Set Loader...')
    train = LiverDataSet(directory=train_folder, augmentation=config['augmentation'],
                         context=config['context'], do_normalize=config['do_normalize'])
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=train.getWeights(),
                                                                   num_samples=config['num_samples'])
    train_data = torch.utils.data.DataLoader(train, batch_size=config['batch_size'], shuffle=False,
                                             sampler=train_sampler, num_workers=config['num_workers'])
    print('Training Loader built!')
    return train_data


def val_data_loader(val_folder, config):
    print('Building Validation Set Loader...')
    val = LiverDataSet(directory=val_folder, context=config['context'])
    val_data_list = []
    patients = val.getPatients()
    for key in patients.keys():
        samples = patients[key]
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(samples)
        val_data = torch.utils.data.DataLoader(val, batch_size=config['batch_size'], shuffle=False,
                                               sampler=val_sampler, num_workers=config['num_workers'])
        val_data_list.append(val_data)
    print('Validation Loader built!')
    return val_data_list

def train_data_loader3d(train_folder, config):
    print('Building Training Set Loader...')
    train = DataLoader3D(directory=train_folder, augmentation=config['augmentation'], context=config['depth'],
                         image_size=config['sample_xy_size'])
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=train.getWeights(),
                                                                   num_samples=config['num_samples'])
    train_data = torch.utils.data.DataLoader(train, batch_size=config['batch_size'], shuffle=False,
                                             sampler=train_sampler, num_workers=config['num_workers'])
    print('Training Loader built!')
    return train_data

def val_data_loader3d(val_folder, config):
    print('Building Validation Set Loader...')
    val = DataLoader3D(directory=val_folder, context=config['depth'], image_size=config['image_size'])
    val_data_list = []
    patients = val.getPatients()
    for key in patients.keys():
        samples = patients[key]
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(samples)
        val_data = torch.utils.data.DataLoader(val, batch_size=config['batch_size'], shuffle=False,
                                               sampler=val_sampler, num_workers=config['num_workers'])
        val_data_list.append(val_data)
    print('Validation Loader built!')
    return val_data_list
