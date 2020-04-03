import torch
import torch.nn as nn
import torch.nn.functional as F


class RegNet(nn.Module):

    def __init__(self, in_channels=5, out_feature=9):
        super(RegNet, self).__init__()

        # input shape: B x in_channels x 512 x 512
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3), padding=1)
        # output shape: B x 32 x 512 x 512
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        # output shape: B x 32 x 256 x 256
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=1)
        # output shape: B x 16 x 256 x 256
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=1)
        # output shape: B x 8 x 256 x 256
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        # output shape: B x 8 x 128 x 128
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 3), padding=1)
        # output shape: B x 4 x 128 x 128
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        # output shape: B x 8 x 64 x 64

        self.fc1 = nn.Linear(in_features=4*64*64, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=out_feature)

    def forward(self, x):

        # Convolutional layers
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Fully connected layers
        x = x.view(-1, 4*64*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# sizes = (1, 5, 512, 512)
# net = RegNet()
# example_in = torch.rand(sizes)
# example_out = net(example_in)
#
# print("Out Shape = ", example_out.shape)
# example_gt = torch.rand((1, 9))
#
# criterion = nn.MSELoss()
# loss = criterion(example_out, example_gt)
#
# print("MSE Loss = ", loss.item())