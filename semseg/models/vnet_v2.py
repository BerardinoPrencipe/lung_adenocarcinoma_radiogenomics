import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_calc import use_multi_gpu_model


# 2D variation of VNet - similar to UNet
# added residual functions to each block
# & down convolutions instead of pooling
# & batch normalization for convolutions
# & drop out before every upsample layer
# & context parameter to make it 2.5 dim


class InitialConv(nn.Module):
    def __init__(self, context, out_channels=16):
        super().__init__()
        self.out_channels = out_channels

        self.conv = nn.Conv2d(in_channels=1 + context * 2, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.conv_down = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=2, stride=2, padding=0)
        self.bn_down = nn.BatchNorm2d(out_channels * 2)

    def forward(self, x):
        layer = F.relu(self.bn(self.conv(x)))
        layer = torch.add(layer, torch.cat([x[:,0:1,:,:]]*self.out_channels, 1))

        conv = F.relu(self.bn_down(self.conv_down(layer)))
        return layer, conv

class DownConvBlock2b(nn.Module):
    def __init__(self, out_channels=32):
        super().__init__()
        self.out_channels = out_channels

        self.conv_a = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_a = nn.BatchNorm2d(out_channels)
        self.conv_b = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_b = nn.BatchNorm2d(out_channels)
        self.conv_down = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=2, stride=2, padding=0)
        self.bn_down = nn.BatchNorm2d(out_channels * 2)

    def forward(self, x):
        layer = F.relu(self.bn_a(self.conv_a(x)))
        layer = F.relu(self.bn_b(self.conv_b(layer)))
        layer = torch.add(layer, x)

        conv = F.relu(self.bn_down(self.conv_down(layer)))
        return layer, conv

class UpConvBlock2b(nn.Module):
    def __init__(self, out_channels=64, undersampling_factor=4):
        super().__init__()
        self.out_channels = out_channels
        self.undersampling_factor = undersampling_factor

        self.conv_a = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_a = nn.BatchNorm2d(out_channels)
        self.conv_b = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_b = nn.BatchNorm2d(out_channels)
        self.conv_up = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels // undersampling_factor, kernel_size=2, stride=2, padding=0)
        self.bn_up = nn.BatchNorm2d(out_channels // undersampling_factor)

    def forward(self, x):
        layer = F.relu(self.bn_a(self.conv_a(x)))
        layer = F.relu(self.bn_b(self.conv_b(layer)))
        layer = torch.add(layer, x)

        conv = F.relu(self.bn_up(self.conv_up(layer)))
        return layer, conv

class DownConvBlock3b(nn.Module):
    def __init__(self, out_channels=64):
        super().__init__()
        self.out_channels = out_channels

        self.conv_a = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_a = nn.BatchNorm2d(out_channels)
        self.conv_b = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_b = nn.BatchNorm2d(out_channels)
        self.conv_c = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_c = nn.BatchNorm2d(out_channels)
        self.conv_down = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=2, stride=2, padding=0)
        self.bn_down = nn.BatchNorm2d(out_channels * 2)

    def forward(self, x):
        layer = F.relu(self.bn_a(self.conv_a(x)))
        layer = F.relu(self.bn_b(self.conv_b(layer)))
        layer = F.relu(self.bn_c(self.conv_c(layer)))
        layer = torch.add(layer, x)

        conv = F.relu(self.bn_down(self.conv_down(layer)))
        return layer, conv


class UpConvBlock3b(nn.Module):
    def __init__(self, out_channels=256, undersampling_factor=2):
        super().__init__()
        self.out_channels = out_channels

        self.conv_a = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_a = nn.BatchNorm2d(out_channels)
        self.conv_b = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_b = nn.BatchNorm2d(out_channels)
        self.conv_c = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_c = nn.BatchNorm2d(out_channels)
        self.conv_up = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels // undersampling_factor, kernel_size=2, stride=2, padding=0)
        self.bn_up = nn.BatchNorm2d(out_channels // undersampling_factor)

    def forward(self, x):
        layer = F.relu(self.bn_a(self.conv_a(x)))
        layer = F.relu(self.bn_b(self.conv_b(layer)))
        layer = F.relu(self.bn_c(self.conv_c(layer)))
        layer = torch.add(layer, x)

        conv = F.relu(self.bn_up(self.conv_up(layer)))
        return layer, conv


class FinalConv(nn.Module):
    def __init__(self, num_outs=2, out_channels=32, no_softmax=True):
        super().__init__()
        self.no_softmax = no_softmax
        self.conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv_1x1 = nn.Conv2d(in_channels=out_channels, out_channels=num_outs, kernel_size=1, stride=1, padding=0)
        self.bn_1x1 = nn.BatchNorm2d(num_outs)
        self.final = F.softmax
    def forward(self, x):
        layer = F.relu(self.bn(self.conv(x)))
        layer = torch.add(layer, x)
        layer = self.bn_1x1(self.conv_1x1(layer))
        if self.no_softmax:
            layer = self.final(layer, dim=1)
        return layer

class CatBlock(nn.Module):
    def __init__(self, dropout=False):
        super().__init__()
        self.dropout = dropout
        if dropout:
            self.do = nn.Dropout2d()

    def forward(self, x1, x2):
        cat = torch.cat((x1,x2), 1)
        if self.dropout: cat = self.do(cat)
        return cat

class VXNet(nn.Module):

    def __init__(self, dropout=False, context=0, num_outs=2, channels=16, no_softmax=False):

        super(VXNet, self).__init__()

        self.init_conv  = InitialConv(context=context, out_channels=channels)

        self.down_block_1 = DownConvBlock2b(out_channels=channels * 2)
        self.down_block_2 = DownConvBlock3b(out_channels=channels * 4)
        self.down_block_3 = DownConvBlock3b(out_channels=channels * 8)

        self.up_block_4   = UpConvBlock3b(out_channels=channels * 16, undersampling_factor=2)
        self.cat_block_4  = CatBlock(dropout)
        self.up_block_3   = UpConvBlock3b(out_channels=channels * 16, undersampling_factor=4)
        self.cat_block_3  = CatBlock(dropout)
        self.up_block_2   = UpConvBlock3b(out_channels=channels * 8, undersampling_factor=4)
        self.cat_block_2  = CatBlock(dropout)
        self.up_block_1   = UpConvBlock2b(out_channels=channels * 4, undersampling_factor=4)
        self.cat_block_1  = CatBlock(dropout)

        self.out_conv     = FinalConv(num_outs=num_outs, no_softmax=no_softmax)


    def forward(self, x):

        layer_down_1, conv_down_1 = self.init_conv(x)
        layer_down_2, conv_down_2 = self.down_block_1(conv_down_1)
        layer_down_3, conv_down_3 = self.down_block_2(conv_down_2)
        layer_down_4, conv_down_4 = self.down_block_3(conv_down_3)

        layer_up_4, conv_up_4 = self.up_block_4(conv_down_4)
        cat4 = self.cat_block_4(conv_up_4, layer_down_4)

        layer_up_3, conv_up_3 = self.up_block_3(cat4)
        cat3 = self.cat_block_3(conv_up_3, layer_down_3)

        layer_up_2, conv_up_2 = self.up_block_2(cat3)
        cat2 = self.cat_block_2(conv_up_2, layer_down_2)

        layer_up_1, conv_up_1 = self.up_block_1(cat2)
        cat1 = self.cat_block_1(conv_up_1, layer_down_1)

        layer_out = self.out_conv(cat1)
        return layer_out

def build_VXNet_with_config(config):
    net = VXNet(dropout=config['dropout'], context=config['context'],
                num_outs=config['num_outs'], no_softmax=config['no_softmax'])
    if config['cuda'] and not config['use_multi_gpu']: net = net.cuda()
    if config['cuda'] and config['use_multi_gpu']: net = use_multi_gpu_model(net)
    return net
