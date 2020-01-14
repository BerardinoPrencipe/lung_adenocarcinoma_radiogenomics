from semseg.models.unet_v2 import UNet as UNetv2
from semseg.models.unet import UNet

n_channels = 5
n_outputs  = 2

net = UNet()
net2 = UNetv2(n_channels, n_outputs)