import torch
from projects.liver.regression.dataset import RegressionSegmentDataset
from projects.liver.geometric.JSONParser import JSONParser

def train_data_loader(train_folder):

    parser = JSONParser("planes.json")
    dataset = RegressionSegmentDataset(parser=parser, directory=train_folder, context=4)

    train_data = torch.utils.data.DataLoader(dataset, batch_size=1)
    return train_data


#train = train_data_loader("/datasets/LiverDecathlon/nii/labels_segments/")