import time

import numpy as np
import torch

from utils_calc import cuda


def map_thickness_to_spacing_context(thickness):
    spacing = int(6 - thickness)
    if spacing < 1:
        print('spacing too low: {}, returning spacing = 1'.format(spacing))
        spacing = 1
    return spacing

def perform_inference_volumetric_image(net, data, context=2, spacing_context=1, do_round=True,
                                       cuda_dev=torch.device('cuda'), do_argmax=False):
    assert do_round is False or do_argmax is False, "do_round={} do_argmax={}".format(do_round, do_argmax)
    assert spacing_context >= 1, "Spacing context must be greater or equal than 1! spacing_context = {}".format(spacing_context)

    start_time = time.time()

    # save output here
    output = np.zeros(data.shape)

    # loop through z-axis
    for i in range(len(data)):

        # append multiple slices in a row
        slices_input = []
        z = i - (context * spacing_context)

        # middle slice first, same as during training
        slices_input.append(np.expand_dims(data[i], 0))

        while z <= i + (context * spacing_context):

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
            z += spacing_context

        inputs = np.expand_dims(np.concatenate(slices_input, 0), 0)

        with torch.no_grad():
            # run slices through the network and save the predictions
            inputs = torch.from_numpy(inputs).float()
            if cuda: inputs = inputs.cuda(cuda_dev)

            # inference
            outputs = net(inputs)
            if do_argmax:
                outputs = torch.argmax(outputs, dim=1)
                outputs = outputs[0,:,:]
            elif do_round:
                outputs = outputs.round()
                outputs = outputs[0, 1, :, :]

            outputs = outputs.data.cpu().numpy()

            output[i, :, :] = outputs

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time is: ", elapsed_time, " for processing image with shape: ", output.shape)
    return output


def perform_inference_volumetric_image_3d(net, data, depth=16, do_round=True,
                                          cuda_dev=torch.device('cuda'), do_argmax=False):
    assert do_round is False or do_argmax is False, "do_round={} do_argmax={}".format(do_round, do_argmax)

    start_time = time.time()

    # save output here
    output = np.zeros(data.shape)

    # loop through z-axis
    idx_start = 0
    for i in range(idx_start, len(data), depth):
        if i+depth >= data.shape[0]:
            i = data.shape[0]-depth

        inputs = data[i:i+depth, :, :]          # 3D
        inputs = np.expand_dims(inputs, axis=0) # 4D
        inputs = np.expand_dims(inputs, axis=0) # 5D
        # DEBUG ONLY
        # print('Inputs   shape = {}'.format(inputs.shape))
        # print('Expected shape = {}'.format((1,1,depth,512,512)))

        with torch.no_grad():
            # run slices through the network and save the predictions
            inputs = torch.from_numpy(inputs).float()
            if cuda: inputs = inputs.cuda(cuda_dev)

            # inference
            outputs = net(inputs)

            if do_argmax:
                outputs = torch.argmax(outputs, dim=1)
                outputs = outputs[0,:,:]
            elif do_round:
                outputs = outputs.round()
                outputs = outputs[0, 1, :, :]

            outputs = outputs.data.cpu().numpy()

            output[i:i+depth, :, :] = outputs

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time is: ", elapsed_time, " for processing image with shape: ", output.shape)
    return output