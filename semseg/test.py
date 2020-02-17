import torch
import torch.nn as nn
import numpy as np
import os

from semseg.models.unet_v2 import UNet as UNetv2
from semseg.models.vnet_v2 import VXNet
from semseg.models.unet import UNet, UNetSmall
from semseg.models.vnet import VNet, VNet_Xtra

from semseg.loss import dice_n_classes, one_hot_encode

''' 
context = 2
n_channels = 5
n_outputs  = 2

net_input_shape = (1,5,512,512)
input_tensor = torch.rand(net_input_shape)

net   = UNet(n_channels, n_outputs)
net2  = UNetv2(n_channels, n_outputs)
net3  = UNetSmall(n_channels, n_outputs)
vnet  = VNet(context=context, num_outs=n_outputs)
vnet2 = VNet_Xtra(context=context, num_outs= n_outputs)

out_1 = net(input_tensor)
out_2 = net2(input_tensor)
out_3 = vnet(input_tensor)
out_4 = vnet2(input_tensor)
out_5 = net3(input_tensor)

outs = [out_1, out_2, out_3, out_4, out_5]

for i, out in enumerate(outs):
    print("Output ", i, " shape = ", out.shape)
'''


#############
### VXNET ###
#############

context = 2
n_channels = 5
net_input_shape = (1,n_channels,512,512)
input_tensor = torch.rand(net_input_shape).cuda()
n_outputs  = 9
net_out_shape = (512, 512)
ground_truth_tensor = torch.ones(net_out_shape)
dropout = True
no_softmax = True
vnet2_ns = VXNet(dropout=dropout, context=context, num_outs=n_outputs, no_softmax=no_softmax)
vnet2_ys = VXNet(dropout=dropout, context=context, num_outs=n_outputs, no_softmax=False)

vnet2_ns = vnet2_ns.cuda()
vnet2_ys = vnet2_ys.cuda()

out_2_ns = vnet2_ns(input_tensor)
out_2_ys = vnet2_ys(input_tensor)

print('Output shape No  Softmax = {}'.format(out_2_ns.shape))
print('Min = ', out_2_ns.min(), ' Max = ', out_2_ns.max())

print('Output shape Yes Softmax = {}'.format(out_2_ys.shape))
print('Min = ', out_2_ys.min(), ' Max = ', out_2_ys.max())

##################
### REAL INPUT ###
##################

path_folder_slices = 'E:/Datasets/LiverDecathlon/npy/train'
central_idx = 26
input_slices_np = np.zeros(input_tensor.shape)
for idx_slices, idx in enumerate(range(central_idx-2, central_idx+2)):
    path_input_slice = os.path.join(path_folder_slices, 'volume-050_{}.npy'.format(central_idx))
    input_slice_np = np.load(path_input_slice)
    input_slices_np[0,idx_slices] = input_slice_np

path_label_slice = os.path.join(path_folder_slices, 'segmentation-050_26.npy')
label_slice_np = np.load(path_label_slice)
label_slice_tensor = torch.from_numpy(label_slice_np.astype(np.uint8)).float().cuda()
label_slice_tensor = label_slice_tensor.squeeze(dim=1)

input_slices_tensor = torch.from_numpy(input_slices_np).float().cuda()
output_slice = vnet2_ys(input_slices_tensor)

print('Output Slice shape = {}'.format(output_slice.shape))
print('Label  Slice shape = {}'.format(label_slice_tensor.shape))

####################################
### DICE N CLASSES in FUNC + OHE ###
####################################
batch_label_slice_tensor = label_slice_tensor.unsqueeze(dim=0)
print('Shape after unsqueeze(): {}'.format(batch_label_slice_tensor.shape))
label_slice_oh_new = one_hot_encode(batch_label_slice_tensor, n_outputs)


################
### TEST OHE ###
################
bb_no_ohe = batch_label_slice_tensor[0, 340:350, 295:305]
bb_ye_ohe = label_slice_oh_new[0, :, 340:350, 295:305]
print(bb_no_ohe)
print(bb_ye_ohe)

print('After  OHE  Shape: {}'.format(label_slice_oh_new.shape))
print('Output VNET Shape: {}'.format(output_slice.shape))
print('Min = ', label_slice_oh_new.min(), ' Max = ', label_slice_oh_new.max())

output_slice = output_slice.cuda()
label_slice_oh_new = label_slice_oh_new.cuda()
dices_new = dice_n_classes(output_slice, label_slice_oh_new)
print('Multi Dice Loss [New OHE] = {}'.format(dices_new))

output_slice_argmax   = torch.argmax(output_slice, dim=1)
label_slice_oh_argmax_new = torch.argmax(label_slice_oh_new, dim=1)

####################################
### DICE N CLASSES + OHE in FUNC ###
####################################
label_slice_tensor = label_slice_tensor.cuda()
dices_ohe = dice_n_classes(output_slice, label_slice_tensor, do_one_hot=True)
print('Multi Dice Loss After OHE = {}'.format(dices_ohe))

outputs, labels = output_slice_argmax, label_slice_tensor
outputs, labels = outputs.cpu().numpy()[0], labels.cpu().numpy()
acc = (outputs == labels).sum() / float(outputs.size)
print('Accuracy                  = {}'.format(acc))

# Debug only
''' 
gt = torch.ones([1,16,16])
pred = torch.ones([1,16,16]) * 0.95

d = dice(pred, gt)
t = tversky(pred, gt)
print('Dice    Loss = {}'.format(d))
print('Tversky Loss = {}'.format(t))
'''
