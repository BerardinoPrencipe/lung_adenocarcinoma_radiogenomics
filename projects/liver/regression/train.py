import torch
import time
import torch.nn as nn
from projects.liver.regression.regression_net import RegNet
from projects.liver.regression.util_dataset import train_data_loader

net = RegNet(in_channels=4)

cuda = torch.cuda.is_available()
if cuda:
    net = net.cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

dataset = train_data_loader("./datasets/LiverDecathlon/nii/images/")

for epoch in range(1):

    start = time.time()

    running_loss = 0
    for i, data in enumerate(dataset, 0):

        input, label = data
        if cuda:
            input, label = input.cuda(), label.cuda()

        optimizer.zero_grad()
        output = net(input)

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

    running_loss += loss.item()
    print("loss: ", running_loss)
    print("Exection time: ", time.time()-start)
# #%% Instantiate a network
# net = get_mobilenet_v2(num_classes=NUM_CLASSES, pretrained=True)
#
# if cuda:
#     net = net.cuda()
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#
# #%% Loop on dataset
# for epoch in range(epochs):  # loop over the dataset multiple times
#
#     start_time = time.time()
#     running_loss = 0.0
#     for i, data in enumerate(train_dataloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data
#         if cuda:
#             inputs, labels = inputs.cuda(), labels.cuda()
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = net(inputs)
#         # K = 6 classes (if we consider only the macro classes)
#         # inputs B x C x H x W
#         # outputs B x K
#
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         # print statistics
#         running_loss += loss.item()
#         if i % steps_loss == (steps_loss-1):    # print every steps_loss mini-batches
#             print("[Epoch {:2d} - Iter {:3d}] loss: {:.3f}".format(epoch + 1, i + 1, running_loss / steps_loss))
#             running_loss = 0.0
#     elapsed_time = time.time() - start_time
#     print("[Epoch {:2d}] elapsed time: {:.3f}".format(epoch+1, elapsed_time) )
#
# print('Finished Training')


