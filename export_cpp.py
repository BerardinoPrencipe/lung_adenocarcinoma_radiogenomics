import os
import networks
import numpy as np
import torch

cuda = torch.cuda.is_available()

# name of the model saved
model_name = '25D'
logs_dir = 'logs'

path_0000_model = os.path.join(logs_dir,"model_epoch_0000.pht")
path_final_model = os.path.join(logs_dir,"model_"+model_name+".pht")

net_input_shape = (1,5,512,512)
example = torch.rand(net_input_shape) # .cuda()

#%%
dropout = True
dice = True
context = 2

net_from_scratch = networks.VNet_Xtra(dice=dice, dropout=dropout, context=context)
if cuda: net_from_scratch = net_from_scratch # .cuda()
state_dict = torch.load(path_0000_model)
net_from_scratch.load_state_dict(state_dict)
tr = torch.jit.script(net_from_scratch)
tr2 = torch.jit.trace(net_from_scratch, example)
#%%
input_gpu = torch.ones(net_input_shape) # .cuda()
output_traced   = tr(input_gpu)
output_traced2   = tr2(input_gpu)
output_original = net_from_scratch(input_gpu)
print("Output TorchScript1 [0,0,0,:5] = ", output_traced[0,0,0,:5])
print("Output TorchScript2 [0,0,0,:5] = ", output_traced2[0,0,0,:5])
print("Output Original     [0,0,0,:5] = ", output_original[0,0,0,:5])
tr.save(os.path.join(logs_dir,"traced_model_"+model_name+".pt"))