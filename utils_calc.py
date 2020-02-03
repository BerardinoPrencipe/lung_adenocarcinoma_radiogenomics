import numpy as np
import torch
import time
import SimpleITK as sitk
import nibabel as nib
import os
cuda = torch.cuda.is_available()

# TODO: deprecated function! Delete and replace with v2
def normalize_data_old(data, dmin=-200, dmax=200):
    return np.clip(data, dmin, dmax) / (dmax-dmin) + 0.5

def normalize_data(data, interval=(-150,350)):
    dmin, dmax = interval
    clipped_data = np.clip(data, dmin, dmax)
    norm_data = (clipped_data - dmin) / (dmax-dmin)
    return norm_data


def perform_inference_volumetric_image(net, data, context=2, do_round=True, cuda_dev=torch.device('cuda')):
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
            outputs = outputs[0, 1, :, :]
            if do_round:
                outputs = outputs.round()
            outputs = outputs.data.cpu().numpy()

            output[i, :, :] = outputs

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time is: ", elapsed_time, " for processing image with shape: ", output.shape)
    return output


def use_multi_gpu_model(net):
    return torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count()))).cuda()


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


def get_patient_id(s):
    return int(s.split("-")[-1].split(".")[0])


def get_dice(y_true, y_pred, smooth=1.):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def get_iou(y_true, y_pred, smooth=1.):
    dice = get_dice(y_true, y_pred, smooth=smooth)
    iou = dice/(2-dice)
    return iou


def normalize(arr, max_value=None):
    if not np.any(arr):
        return arr
    else:
        arr_min = np.min(arr)
        arr_max = np.max(arr) if max_value is None else max_value
        return (arr-arr_min)/(arr_max-arr_min)

def get_acc(tp,tn,fp,fn):
    acc = (tp+tn) / (tp+tn+fp+fn)
    return acc

def get_mcc(tp, tn, fp, fn):
    if tp+fp == 0 or tp+fn==0 or tn+fp==0 or tn+fn==0:
        den = 1
    else:
        den = (np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    mcc = ((tp * tn) - (fp * fn)) / den
    return mcc

def get_dsc(tp, fp, fn):
    dsc = 2*tp / (fp+fn+2*tp)
    return dsc

def get_jac(tp, fp, fn):
    iou = tp / (tp+fp+fn)
    return iou

def get_sens(tp, fn):
    sens = tp / (tp + fn)  # recall
    return sens

def get_spec(tn, fp):
    spec = tn / (tn + fp)
    return spec

def get_confusion_matrix_metrics(tp, tn, fp, fn):
    metrics = {
        'acc' : get_acc(tp=tp, tn=tn, fp=fp, fn=fn),
        'mcc' : get_mcc(tp=tp, tn=tn, fp=fp, fn=fn),
        'dsc' : get_dsc(tp=tp, fp=fp, fn=fn),
        'sens': get_sens(tp=tp, fn=fn),
        'spec': get_spec(tn=tn, fp=fp)
    }
    return metrics