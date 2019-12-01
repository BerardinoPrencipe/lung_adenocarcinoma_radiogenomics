import os
import networks
import numpy as np
import torch
from networks import VNet_Xtra

# name of the model saved
model_name = '25D'
logs_dir = 'logs'
path_final_model = os.path.join(logs_dir,"model_"+model_name+".pht")
undersampling_factor = 4
height = 1280 // undersampling_factor
width = 960 // undersampling_factor
net_input_shape = (1,3,height,width)

#%%
dropout = True
dice = True
context = 1

with torch.no_grad():
    eps = 1e-6
    example = torch.rand(net_input_shape)
    net = VNet_Xtra(context=context)
    net.eval()
    tr = torch.jit.trace(net, example)
    input_gpu = torch.rand(net_input_shape)

    output_traced = tr(input_gpu)
    output_original = net(input_gpu)
    print("Output TorchScript1 [0,0,0,:5]  = ", output_traced[0,0,0,:5])
    print("Output Original     [0,0,0,:5]  = ", output_original[0,0,0,:5])
    print("Identical elements traced 1     = ", (output_traced == output_original).sum())
    print("Elements withing range traced 1 = ",( ( output_traced < (output_original + eps) ) * ((output_original - eps) < output_traced ) ).sum())
    print("Total elements                  = ", 512 * 512 * 2)


traced_model_name = "traced_model_ecodom_"+model_name+".pt"

tr.save(os.path.join(logs_dir,traced_model_name))