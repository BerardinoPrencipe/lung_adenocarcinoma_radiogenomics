import torch
from semseg.models.vnet_v3 import build_VXNet_with_config
from projects.liver.data_util.data_load_util import train_data_loader3d, val_data_loader3d
from projects.liver.train.config import config, get_train_val_folders
from projects.liver.data_util.data_load_3d import DataLoader3D
from semseg.loss import dice
device = torch.device('cuda')

config['batch_size'] = 1
config['use_3d'] = True

num_outs = 2
no_softmax = False
net = build_VXNet_with_config(config)
net = net.cuda(device=device)
net.eval()

train_folder, val_folder = get_train_val_folders(dataset='vessels')

train_data = train_data_loader3d(train_folder=train_folder,config=config)
validation = val_data_loader3d(val_folder=val_folder, config=config)
# B = 1
# C = 1
# D = 16
# H = W = 128
# input_shape = (B,C,D,H,W)
#
# input_tensor = torch.rand(input_shape).cuda(device=device)
# output_tensor = net(input_tensor)
# print('Output Tensor Shape = {}'.format(output_tensor.shape))

net.train()
for i, data in enumerate(train_data):
    image, label = data[0].cuda(device=device), data[1].cuda(device=device)
    out = net(image)
    out = out[:,1].unsqueeze(dim=1)
    dice_loss = dice(out, label)
    print('Output Tensor Shape = {}'.format(out.shape))
    dice_loss.backward()
    if i == 1:
        break

net.eval()
with torch.no_grad():
    for val in validation:
        for i, data in enumerate(val):
            # image, label = data[0].cuda(device=device), data[1].cuda(device=device)
            # out = net(image)
            # out = out[:, 1].unsqueeze(dim=1)
            # dice_loss = dice(out, label)
            # print('Output Tensor Shape = {}'.format(out.shape))
            print(i)
