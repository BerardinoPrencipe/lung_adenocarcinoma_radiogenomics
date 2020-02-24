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
    def __init__(self, context, out_channels=16, use_3d=False):
        super().__init__()
        self.out_channels = out_channels
        bn, conv = get_batchnd(use_3d), get_convnd(use_3d)
        in_channels = 1 if use_3d else 1 + context * 2
        self.conv = conv(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn = bn(out_channels)
        self.conv_down = conv(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=2, stride=2,
                              padding=0)
        self.bn_down = bn(out_channels * 2)


    def forward(self, x):
        layer = F.relu(self.bn(self.conv(x)))
        layer = torch.add(layer, torch.cat([x[:,0:1]]*self.out_channels, 1))

        conv = F.relu(self.bn_down(self.conv_down(layer)))
        return layer, conv

class DownConvBlock2b(nn.Module):
    def __init__(self, out_channels=32, use_3d=False):
        super().__init__()
        self.out_channels = out_channels
        bn, conv = get_batchnd(use_3d), get_convnd(use_3d)

        self.conv_a = conv(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_a = bn(out_channels)
        self.conv_b = conv(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_b = bn(out_channels)
        self.conv_down = conv(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=2, stride=2, padding=0)
        self.bn_down = bn(out_channels * 2)

    def forward(self, x):
        layer = F.relu(self.bn_a(self.conv_a(x)))
        layer = F.relu(self.bn_b(self.conv_b(layer)))
        layer = torch.add(layer, x)

        conv = F.relu(self.bn_down(self.conv_down(layer)))
        return layer, conv

class UpConvBlock2b(nn.Module):
    def __init__(self, out_channels=64, undersampling_factor=4, use_3d=False):
        super().__init__()
        self.out_channels = out_channels
        self.undersampling_factor = undersampling_factor
        bn, conv, conv_transpose = get_batchnd(use_3d), get_convnd(use_3d), get_convtrannd(use_3d)

        self.conv_a = conv(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_a = bn(out_channels)
        self.conv_b = conv(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_b = bn(out_channels)
        self.conv_up = conv_transpose(in_channels=out_channels, out_channels=out_channels // undersampling_factor, kernel_size=2, stride=2, padding=0)
        self.bn_up = bn(out_channels // undersampling_factor)

    def forward(self, x):
        layer = F.relu(self.bn_a(self.conv_a(x)))
        layer = F.relu(self.bn_b(self.conv_b(layer)))
        layer = torch.add(layer, x)

        conv = F.relu(self.bn_up(self.conv_up(layer)))
        return layer, conv

class DownConvBlock3b(nn.Module):
    def __init__(self, out_channels=64, use_3d=False):
        super().__init__()
        self.out_channels = out_channels
        bn, conv = get_batchnd(use_3d), get_convnd(use_3d)

        self.conv_a = conv(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_a = bn(out_channels)
        self.conv_b = conv(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_b = bn(out_channels)
        self.conv_c = conv(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_c = bn(out_channels)
        self.conv_down = conv(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=2, stride=2, padding=0)
        self.bn_down = bn(out_channels * 2)

    def forward(self, x):
        layer = F.relu(self.bn_a(self.conv_a(x)))
        layer = F.relu(self.bn_b(self.conv_b(layer)))
        layer = F.relu(self.bn_c(self.conv_c(layer)))
        layer = torch.add(layer, x)

        conv = F.relu(self.bn_down(self.conv_down(layer)))
        return layer, conv


class UpConvBlock3b(nn.Module):
    def __init__(self, out_channels=256, undersampling_factor=2, use_3d=False):
        super().__init__()
        self.out_channels = out_channels
        bn, conv, conv_transpose = get_batchnd(use_3d), get_convnd(use_3d), get_convtrannd(use_3d)

        self.conv_a = conv(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_a = bn(out_channels)
        self.conv_b = conv(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_b = bn(out_channels)
        self.conv_c = conv(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn_c = bn(out_channels)
        self.conv_up = conv_transpose(in_channels=out_channels, out_channels=out_channels // undersampling_factor, kernel_size=2, stride=2, padding=0)
        self.bn_up = bn(out_channels // undersampling_factor)

    def forward(self, x):
        layer = F.relu(self.bn_a(self.conv_a(x)))
        layer = F.relu(self.bn_b(self.conv_b(layer)))
        layer = F.relu(self.bn_c(self.conv_c(layer)))
        layer = torch.add(layer, x)

        conv = F.relu(self.bn_up(self.conv_up(layer)))
        return layer, conv


class FinalConv(nn.Module):
    def __init__(self, num_outs=2, out_channels=32, no_softmax=True, use_3d=False):
        super().__init__()
        self.no_softmax = no_softmax
        bn, conv = get_batchnd(use_3d), get_convnd(use_3d)

        self.conv = conv(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn = bn(out_channels)
        self.conv_1x1 = conv(in_channels=out_channels, out_channels=num_outs, kernel_size=1, stride=1, padding=0)
        self.bn_1x1 = bn(num_outs)
        self.final = F.softmax
    def forward(self, x):
        layer = F.relu(self.bn(self.conv(x)))
        layer = torch.add(layer, x)
        layer = self.bn_1x1(self.conv_1x1(layer))
        if not self.no_softmax:
            layer = self.final(layer, dim=1)
        return layer

class CatBlock(nn.Module):
    def __init__(self, dropout=False, use_3d=False):
        super().__init__()
        self.dropout = dropout
        drop = get_dropout(use_3d)

        if dropout:
            self.do = drop()

    def forward(self, x1, x2):
        cat = torch.cat((x1,x2), 1)
        if self.dropout: cat = self.do(cat)
        return cat

class VXNet(nn.Module):

    def __init__(self, dropout=False, context=0, num_outs=2, channels=16, no_softmax=False, use_3d=False):

        super(VXNet, self).__init__()


        self.init_conv  = InitialConv(context=context, out_channels=channels, use_3d=use_3d)

        self.down_block_1 = DownConvBlock2b(out_channels=channels * 2, use_3d=use_3d)
        self.down_block_2 = DownConvBlock3b(out_channels=channels * 4, use_3d=use_3d)
        self.down_block_3 = DownConvBlock3b(out_channels=channels * 8, use_3d=use_3d)


        self.up_block_4   = UpConvBlock3b(out_channels=channels * 16, undersampling_factor=2, use_3d=use_3d)
        self.cat_block_4  = CatBlock(dropout, use_3d=use_3d)
        self.up_block_3   = UpConvBlock3b(out_channels=channels * 16, undersampling_factor=4, use_3d=use_3d)
        self.cat_block_3  = CatBlock(dropout, use_3d=use_3d)
        self.up_block_2   = UpConvBlock3b(out_channels=channels * 8, undersampling_factor=4, use_3d=use_3d)
        self.cat_block_2  = CatBlock(dropout, use_3d=use_3d)
        self.up_block_1   = UpConvBlock2b(out_channels=channels * 4, undersampling_factor=4, use_3d=use_3d)
        self.cat_block_1  = CatBlock(dropout, use_3d=use_3d)

        self.out_conv     = FinalConv(num_outs=num_outs, no_softmax=no_softmax, use_3d=use_3d)


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
    if config['use_3d']:
        context = config['depth']
    else:
        context = config['context']
    net = VXNet(dropout=config['dropout'], context=context,
                num_outs=config['num_outs'], no_softmax=config['no_softmax'], use_3d=config['use_3d'])
    if config['cuda'] and not config['use_multi_gpu']: net = net.cuda()
    if config['cuda'] and config['use_multi_gpu']: net = use_multi_gpu_model(net)
    return net


def get_convnd(use_3d):
    return nn.Conv3d if use_3d else nn.Conv2d


def get_convtrannd(use_3d):
    return nn.ConvTranspose3d if use_3d else nn.ConvTranspose2d


def get_batchnd(use_3d):
    return nn.BatchNorm3d if use_3d else nn.BatchNorm2d


def get_dropout(use_3d):
    return nn.Dropout3d if use_3d else nn.Dropout2d