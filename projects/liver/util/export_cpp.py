import os
import numpy as np
import torch

cuda = torch.cuda.is_available()
use_multi = False

# name of the model saved
model_name = '25D'
logs_dir = 'logs'

if not use_multi:
    path_final_model = os.path.join(logs_dir,"model_"+model_name+".pht")
else:
    path_final_model = os.path.join(logs_dir,"multi_gpu","model_"+model_name+".pht")

path_final_model = 'logs/vessels/model_25D__2020-02-01__16_42_49.pht'
path_final_model = 'logs/vessels/model_25D__2020-02-02__18_15_53.pht'
path_final_model = 'logs/vessels_tumors/model_25D__2020-02-20__06_53_17.pht'

from semseg.models.vnet_v2 import VXNet
path_model = 'logs/vessels/model_epoch_0999.pht'

path_final_model = 'logs/segments/model_25D__2020-02-19__07_13_36.pht'

path_final_model = 'logs/liver_no_norm/model_25D__2020-03-05__11_22_13.pht'

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
    # net = VXNet(dropout=True, context=2, num_outs=2, no_softmax=False)
    # net.load_state_dict(torch.load(path_model))

    if cuda: net = net.cuda()
    else: net = net.cpu()
    net.eval()
    if use_multi:
        tr = torch.jit.trace(net.module, example)
    else:
        tr = torch.jit.trace(net, example)
    input_gpu = torch.rand(net_input_shape)
    if cuda: input_gpu = input_gpu.cuda()

    output_traced = tr(input_gpu)
    output_original = net(input_gpu)
    print("Output TorchScript1 [0,0,0,:5]  = ", output_traced[0,0,0,:5])
    print("Output Original     [0,0,0,:5]  = ", output_original[0,0,0,:5])
    print("Identical elements traced 1     = ", (output_traced == output_original).sum())
    print("Elements withing range traced 1 = ",( ( output_traced < (output_original + eps) ) * ((output_original - eps) < output_traced ) ).sum())
    print("Total elements                  = ", 512 * 512 * 5)

if cuda:
    if use_multi:
        traced_model_name = "traced_model_multi_"+model_name+".pt"
    else:
        traced_model_name = "traced_model_"+model_name+".pt"
else:
    traced_model_name = "traced_model_cpu_"+model_name+".pt"
tr.save(os.path.join(logs_dir,traced_model_name))