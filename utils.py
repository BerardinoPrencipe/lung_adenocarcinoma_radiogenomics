import numpy as np
import torch

cuda = torch.cuda.is_available()


def normalize_data(data, dmin=-200, dmax=200):
    return np.clip(data, dmin, dmax) / (dmax-dmin) + 0.5


def perform_inference_volumetric_image(net, data, context=2):
    # save output here
    output = np.zeros((len(data), 512, 512))

    # loop through z-axis
    for i in range(len(data)):

        # append multiple slices in a row
        slices_input = []
        z = i - context

        # middle slice first, same as during training
        slices_input.append(np.expand_dims(data[i], 0))

        while z <= i + context:

            if z == i:
                # middle slice is already appended
                pass
            elif z < 0:
                # append first slice if z falls outside of data bounds
                slices_input.append(np.expand_dims(data[0], 0))
            elif z >= len(data):
                # append last slice if z falls outside of data bounds
                slices_input.append(np.expand_dims(data[len(data) - 1], 0))
            else:
                # append slice z
                slices_input.append(np.expand_dims(data[z], 0))
            z += 1

        inputs = np.expand_dims(np.concatenate(slices_input, 0), 0)

        with torch.no_grad():
            # run slices through the network and save the predictions
            inputs = torch.from_numpy(inputs).float()
            if cuda: inputs = inputs.cuda()

            # inference
            outputs = net(inputs)
            outputs = outputs[0, 1, :, :].round()
            outputs = outputs.data.cpu().numpy()

            output[i, :, :] = outputs
    return output


def use_multi_gpu_model(net):
    return torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count()))).cuda()
