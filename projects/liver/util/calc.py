import time

import SimpleITK as sitk
import numpy as np
import torch

from utils_calc import cuda


def perform_inference_volumetric_image(net, data, context=2, do_round=True,
                                       cuda_dev=torch.device('cuda'), do_argmax=False):
    assert do_round is False or do_argmax is False, "do_round={} do_argmax={}".format(do_round, do_argmax)

    start_time = time.time()

    # save output here
    output = np.zeros(data.shape)

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


def post_process_liver(output, vector_radius=(25, 25, 25), kernel=sitk.sitkBall):
    # Get connected components
    ccif = sitk.ConnectedComponentImageFilter()
    sitk_output = sitk.GetImageFromArray(output)
    conn_comps = ccif.Execute(sitk_output)
    conn_comps_np = sitk.GetArrayFromImage(conn_comps)

    unique_values = np.unique(conn_comps_np)
    n_uniques = len(unique_values)
    counter_uniques = np.zeros(n_uniques)
    for i in range(1, max(unique_values) + 1):
        counter_uniques[i] = (conn_comps_np == i).sum()
    biggest_region_value = np.argmax(counter_uniques)

    # Get largest connected component
    largest_conn_comp = np.zeros(conn_comps_np.shape, dtype=np.uint8)
    largest_conn_comp[conn_comps_np == biggest_region_value] = 1

    # Morphological Closing
    largest_conn_comp_uint8 = largest_conn_comp.astype(np.uint8)
    largest_conn_comp_sitk = sitk.GetImageFromArray(largest_conn_comp_uint8)
    largest_conn_comp_closed_sitk = sitk.BinaryMorphologicalClosing(largest_conn_comp_sitk, vector_radius, kernel)
    largest_conn_comp_closed_np = sitk.GetArrayFromImage(largest_conn_comp_closed_sitk)

    # Output
    print("Liver Voxels in Original Output    : ", output.sum())
    print("Liver Voxels after ConnectCompLabel: ", largest_conn_comp.sum())
    print("Liver Voxels after Closing         : ", largest_conn_comp_closed_np.sum())

    return largest_conn_comp_closed_np

# M[I,J] = 1 if the segment i and j can be in the same slice, otherwise 0
segments_consistency_matrix = np.resize(np.array([1,1,1,1,1,1,1,1,
                                                    1,1,0,1,0,0,1,1,
                                                    1,0,1,1,1,1,1,1,
                                                    1,1,1,1,1,1,1,1,
                                                    1,0,1,1,1,1,0,0,
                                                    1,0,1,1,1,1,0,0,
                                                    1,1,1,1,0,0,1,1,
                                                    1,1,1,1,0,0,1,1]),(8,8))

def consistency_check(slice):
    slice = np.array(slice,dtype=np.uint8).flatten()
    labels = np.unique(slice)
    for idx_i, i in enumerate(labels):
        for j in labels[idx_i+1:]:
            print(i, j)
            if segments_consistency_matrix[i-1,j-1] == 0:

                return False
    return True

consistency_check([1,2,4])
consistency_check([1,6,7])
consistency_check([1,7,7,8,8])
consistency_check([1,7,8,6])