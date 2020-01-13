import torch
from projects.liver.data_util.data_load import LiverDataSet


def train_data_loader(train_folder, config):
    print('Building Training Set Loader...')
    train = LiverDataSet(directory=train_folder, augment=config['augment'], context=config['context'])
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
