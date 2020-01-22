import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from utils import use_multi_gpu_model

# 2D variation of VNet - similar to UNet
# added residual functions to each block
# & down convolutions instead of pooling

class VNet(nn.Module):

    def __init__(self, dice=False, context=0, num_outs=2):

        super(VNet, self).__init__()

        self.conv1 = nn.Conv2d(1 + context * 2, 16, 5, stride=1, padding=2)
        self.conv1_down = nn.Conv2d(16, 32, 2, stride=2, padding=0)

        self.conv2a = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.conv2b = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.conv2_down = nn.Conv2d(32, 64, 2, stride=2, padding=0)

        self.conv3a = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.conv3b = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.conv3c = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.conv3_down = nn.Conv2d(64, 128, 2, stride=2, padding=0)

        self.conv4a = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.conv4b = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.conv4c = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.conv4_down = nn.Conv2d(128, 256, 2, stride=2, padding=0)

        self.conv5a = nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.conv5b = nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.conv5c = nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.conv5_up = nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0)

        self.conv6a = nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.conv6b = nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.conv6c = nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.conv6_up = nn.ConvTranspose2d(256, 64, 2, stride=2, padding=0)

        self.conv7a = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.conv7b = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.conv7c = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.conv7_up = nn.ConvTranspose2d(128, 32, 2, stride=2, padding=0)

        self.conv8a = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.conv8b = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.conv8_up = nn.ConvTranspose2d(64, 16, 2, stride=2, padding=0)

        self.conv9 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.conv9_1x1 = nn.Conv2d(32, num_outs, 1, stride=1, padding=0)

        if dice:
            self.final = F.softmax
        else:
            self.final = F.log_softmax

    def switch(self, dice):

        if dice:
            self.final = F.softmax
        else:
            self.final = F.log_softmax

    def forward(self, x):

        layer1 = F.relu(self.conv1(x))
        layer1 = torch.add(layer1, torch.cat([x[:, 0:1, :, :]] * 16, 1))

        conv1 = F.relu(self.conv1_down(layer1))

        layer2 = F.relu(self.conv2a(conv1))
        layer2 = F.relu(self.conv2b(layer2))
        layer2 = torch.add(layer2, conv1)

        conv2 = F.relu(self.conv2_down(layer2))

        layer3 = F.relu(self.conv3a(conv2))
        layer3 = F.relu(self.conv3b(layer3))
        layer3 = F.relu(self.conv3c(layer3))
        layer3 = torch.add(layer3, conv2)

        conv3 = F.relu(self.conv3_down(layer3))

        layer4 = F.relu(self.conv4a(conv3))
        layer4 = F.relu(self.conv4b(layer4))
        layer4 = F.relu(self.conv4c(layer4))
        layer4 = torch.add(layer4, conv3)

        conv4 = F.relu(self.conv4_down(layer4))

        layer5 = F.relu(self.conv5a(conv4))
        layer5 = F.relu(self.conv5b(layer5))
        layer5 = F.relu(self.conv5c(layer5))
        layer5 = torch.add(layer5, conv4)

        conv5 = F.relu(self.conv5_up(layer5))

        cat6 = torch.cat((conv5, layer4), 1)

        layer6 = F.relu(self.conv6a(cat6))
        layer6 = F.relu(self.conv6b(layer6))
        layer6 = F.relu(self.conv6c(layer6))
        layer6 = torch.add(layer6, cat6)

        conv6 = F.relu(self.conv6_up(layer6))

        cat7 = torch.cat((conv6, layer3), 1)

        layer7 = F.relu(self.conv7a(cat7))
        layer7 = F.relu(self.conv7b(layer7))
        layer7 = F.relu(self.conv7c(layer7))
        layer7 = torch.add(layer7, cat7)

        conv7 = F.relu(self.conv7_up(layer7))

        cat8 = torch.cat((conv7, layer2), 1)

        layer8 = F.relu(self.conv8a(cat8))
        layer8 = F.relu(self.conv8b(layer8))
        layer8 = torch.add(layer8, cat8)

        conv8 = F.relu(self.conv8_up(layer8))

        cat9 = torch.cat((conv8, layer1), 1)

        layer9 = F.relu(self.conv9(cat9))
        layer9 = torch.add(layer9, cat9)
        layer9 = self.final(self.conv9_1x1(layer9), dim=1)

        return layer9


# 2D variation of VNet - similar to UNet
# added residual functions to each block
# & down convolutions instead of pooling
# & batch normalization for convolutions
# & drop out before every upsample layer
# & context parameter to make it 2.5 dim

class VNet_Xtra(nn.Module):

    def __init__(self, dice=False, dropout=False, context=0, num_outs=2):

        super(VNet_Xtra, self).__init__()

        self.dropout = dropout
        if self.dropout:
            self.do6 = nn.Dropout2d()
            self.do7 = nn.Dropout2d()
            self.do8 = nn.Dropout2d()
            self.do9 = nn.Dropout2d()

        self.conv1 = nn.Conv2d(1 + context * 2, 16, 5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv1_down = nn.Conv2d(16, 32, 2, stride=2, padding=0)
        self.bn1_down = nn.BatchNorm2d(32)

        self.conv2a = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.bn2a = nn.BatchNorm2d(32)
        self.conv2b = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.bn2b = nn.BatchNorm2d(32)
        self.conv2_down = nn.Conv2d(32, 64, 2, stride=2, padding=0)
        self.bn2_down = nn.BatchNorm2d(64)

        self.conv3a = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.bn3a = nn.BatchNorm2d(64)
        self.conv3b = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.bn3b = nn.BatchNorm2d(64)
        self.conv3c = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.bn3c = nn.BatchNorm2d(64)
        self.conv3_down = nn.Conv2d(64, 128, 2, stride=2, padding=0)
        self.bn3_down = nn.BatchNorm2d(128)

        self.conv4a = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.bn4a = nn.BatchNorm2d(128)
        self.conv4b = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.bn4b = nn.BatchNorm2d(128)
        self.conv4c = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.bn4c = nn.BatchNorm2d(128)
        self.conv4_down = nn.Conv2d(128, 256, 2, stride=2, padding=0)
        self.bn4_down = nn.BatchNorm2d(256)

        self.conv5a = nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.bn5a = nn.BatchNorm2d(256)
        self.conv5b = nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.bn5b = nn.BatchNorm2d(256)
        self.conv5c = nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.bn5c = nn.BatchNorm2d(256)
        self.conv5_up = nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0)
        self.bn5_up = nn.BatchNorm2d(128)

        self.conv6a = nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.bn6a = nn.BatchNorm2d(256)
        self.conv6b = nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.bn6b = nn.BatchNorm2d(256)
        self.conv6c = nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.bn6c = nn.BatchNorm2d(256)
        self.conv6_up = nn.ConvTranspose2d(256, 64, 2, stride=2, padding=0)
        self.bn6_up = nn.BatchNorm2d(64)

        self.conv7a = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.bn7a = nn.BatchNorm2d(128)
        self.conv7b = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.bn7b = nn.BatchNorm2d(128)
        self.conv7c = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.bn7c = nn.BatchNorm2d(128)
        self.conv7_up = nn.ConvTranspose2d(128, 32, 2, stride=2, padding=0)
        self.bn7_up = nn.BatchNorm2d(32)

        self.conv8a = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.bn8a = nn.BatchNorm2d(64)
        self.conv8b = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.bn8b = nn.BatchNorm2d(64)
        self.conv8_up = nn.ConvTranspose2d(64, 16, 2, stride=2, padding=0)
        self.bn8_up = nn.BatchNorm2d(16)

        self.conv9 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.bn9 = nn.BatchNorm2d(32)
        self.conv9_1x1 = nn.Conv2d(32, num_outs, 1, stride=1, padding=0)
        self.bn9_1x1 = nn.BatchNorm2d(num_outs)

        if dice:
            self.final = F.softmax
        else:
            self.final = F.log_softmax

    def switch(self, dice):

        if dice:
            self.final = F.softmax
        else:
            self.final = F.log_softmax

    def forward(self, x):

        layer1 = F.relu(self.bn1(self.conv1(x)))
        layer1 = torch.add(layer1, torch.cat([x[:, 0:1, :, :]] * 16, 1))

        conv1 = F.relu(self.bn1_down(self.conv1_down(layer1)))

        layer2 = F.relu(self.bn2a(self.conv2a(conv1)))
        layer2 = F.relu(self.bn2b(self.conv2b(layer2)))
        layer2 = torch.add(layer2, conv1)

        conv2 = F.relu(self.bn2_down(self.conv2_down(layer2)))

        layer3 = F.relu(self.bn3a(self.conv3a(conv2)))
        layer3 = F.relu(self.bn3b(self.conv3b(layer3)))
        layer3 = F.relu(self.bn3c(self.conv3c(layer3)))
        layer3 = torch.add(layer3, conv2)

        conv3 = F.relu(self.bn3_down(self.conv3_down(layer3)))

        layer4 = F.relu(self.bn4a(self.conv4a(conv3)))
        layer4 = F.relu(self.bn4b(self.conv4b(layer4)))
        layer4 = F.relu(self.bn4c(self.conv4c(layer4)))
        layer4 = torch.add(layer4, conv3)

        conv4 = F.relu(self.bn4_down(self.conv4_down(layer4)))

        layer5 = F.relu(self.bn5a(self.conv5a(conv4)))
        layer5 = F.relu(self.bn5b(self.conv5b(layer5)))
        layer5 = F.relu(self.bn5c(self.conv5c(layer5)))
        layer5 = torch.add(layer5, conv4)

        conv5 = F.relu(self.bn5_up(self.conv5_up(layer5)))

        cat6 = torch.cat((conv5, layer4), 1)

        if self.dropout: cat6 = self.do6(cat6)

        layer6 = F.relu(self.bn6a(self.conv6a(cat6)))
        layer6 = F.relu(self.bn6b(self.conv6b(layer6)))
        layer6 = F.relu(self.bn6c(self.conv6c(layer6)))
        layer6 = torch.add(layer6, cat6)

        conv6 = F.relu(self.bn6_up(self.conv6_up(layer6)))

        cat7 = torch.cat((conv6, layer3), 1)

        if self.dropout: cat7 = self.do7(cat7)

        layer7 = F.relu(self.bn7a(self.conv7a(cat7)))
        layer7 = F.relu(self.bn7b(self.conv7b(layer7)))
        layer7 = F.relu(self.bn7c(self.conv7c(layer7)))
        layer7 = torch.add(layer7, cat7)

        conv7 = F.relu(self.bn7_up(self.conv7_up(layer7)))

        cat8 = torch.cat((conv7, layer2), 1)

        if self.dropout: cat8 = self.do8(cat8)

        layer8 = F.relu(self.bn8a(self.conv8a(cat8)))
        layer8 = F.relu(self.bn8b(self.conv8b(layer8)))
        layer8 = torch.add(layer8, cat8)

        conv8 = F.relu(self.bn8_up(self.conv8_up(layer8)))

        cat9 = torch.cat((conv8, layer1), 1)

        if self.dropout: cat9 = self.do9(cat9)

        layer9 = F.relu(self.bn9(self.conv9(cat9)))
        layer9 = torch.add(layer9, cat9)
        layer9 = self.final(self.bn9_1x1(self.conv9_1x1(layer9)), dim=1)

        return layer9


def build_VNet_Xtra_with_config(config, criterion):
    dice = criterion is None
    net = VNet_Xtra(dice, config['dropout'], config['context'], config['num_outs'])
    if config['cuda'] and not config['use_multi_gpu']: net = net.cuda()
    if config['cuda'] and config['use_multi_gpu']: net = use_multi_gpu_model(net)
    return net