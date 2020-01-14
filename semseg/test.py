import torch

from semseg.models.unet_v2 import UNet as UNetv2
from semseg.models.unet import UNet, UNetSmall
from semseg.models.vnet import VNet, VNet_Xtra

context = 2
n_channels = 5
n_outputs  = 2

net   = UNet(n_channels, n_outputs)
net2  = UNetv2(n_channels, n_outputs)
net3  = UNetSmall(n_channels, n_outputs)
vnet  = VNet(context=context, num_outs=n_outputs)
vnet2 = VNet_Xtra(context=context, num_outs= n_outputs)

net_input_shape = (1,5,512,512)
input_tensor = torch.rand(net_input_shape)

out_1 = net(input_tensor)
out_2 = net2(input_tensor)
out_3 = vnet(input_tensor)
out_4 = vnet2(input_tensor)
out_5 = net3(input_tensor)

outs = [out_1, out_2, out_3, out_4, out_5]

for i, out in enumerate(outs):
    print("Output ", i, " shape = ", out.shape)
