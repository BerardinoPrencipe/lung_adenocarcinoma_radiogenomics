import os
import networks
import numpy as np
import torch

cuda = torch.cuda.is_available()

# name of the model saved
model_name = '25D'
logs_dir = 'logs'

path_final_model = os.path.join(logs_dir,"model_"+model_name+".pht")

net_input_shape = (1,5,512,512)


#%%
dropout = True
dice = True
context = 2

with torch.no_grad():
    eps = 1e-6
    example = torch.rand(net_input_shape)
    if cuda: example = example.cuda()
    net = torch.load(path_final_model)
    if cuda: net = net.cuda()
    else: net = net.cpu()
    net.eval()
    tr = torch.jit.trace(net, example)
    input_gpu = torch.rand(net_input_shape)
    if cuda: input_gpu = input_gpu.cuda()

    output_traced = tr(input_gpu)
    output_original = net(input_gpu)
    print("Output TorchScript1 [0,0,0,:5] = ", output_traced[0,0,0,:5])
    print("Output Original     [0,0,0,:5] = ", output_original[0,0,0,:5])
    print("Identical elements traced 1    = ", (output_traced == output_original).sum())
    print("Elements withing range traced 1= ",( ( output_traced < (output_original + eps) ) * ((output_original - eps) < output_traced ) ).sum())
    print("Total elements                 = ", 512 * 512 * 2)

if cuda:
    traced_model_name = "traced_model_"+model_name+".pt"
else:
    traced_model_name = "traced_model_cpu_"+model_name+".pt"
tr.save(os.path.join(logs_dir,traced_model_name))