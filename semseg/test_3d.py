import torch
from semseg.models.vnet3d import VXNet

device = torch.device('cuda')

num_outs = 9
no_softmax = False
net = VXNet(num_outs=num_outs, no_softmax=no_softmax)
net = net.cuda(device=device)
net.eval()

B = 1
C = 1
D = 16
H = W = 128
input_shape = (B,C,D,H,W)

input_tensor = torch.rand(input_shape).cuda(device=device)

output_tensor = net(input_tensor)
print('Output Tensor Shape = {}'.format(output_tensor.shape))