import torch
import torchvision

with torch.no_grad():
    model = torchvision.models.resnet18().cuda()

    example = torch.rand(1,3,224,224).cuda()

    tr = torch.jit.trace(model,example)

    example_input = torch.ones(1,3,224,224).cuda()
    out_tr = tr(example_input)
    out_or = model(example_input)

    print('Out TR: ', out_tr[0,:5])
    print('Out OR: ', out_or[0,:5])