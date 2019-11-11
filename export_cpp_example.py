import torch
import torchvision

model = torchvision.models.resnet18()

example = torch.rand(1,3,224,224)

tr = torch.jit.trace(model,example)

out_tr = tr(torch.ones(1,3,224,224))
out_or = model(torch.ones(1,3,224,224))

print('Out TR: ', out_tr[0,:5])
print('Out OR: ', out_or[0,:5])