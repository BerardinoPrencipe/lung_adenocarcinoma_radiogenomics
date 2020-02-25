import numpy as np
import torch

cuda = torch.cuda.is_available()

# TODO: deprecated function! Delete and replace with v2
def normalize_data_old(data, dmin=-200, dmax=200):
    return np.clip(data, dmin, dmax) / (dmax-dmin) + 0.5

def normalize_data(data, interval=(-150,350)):
    dmin, dmax = interval
    clipped_data = np.clip(data, dmin, dmax)
    norm_data = (clipped_data - dmin) / (dmax-dmin)
    return norm_data



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

  
def use_multi_gpu_model(net):
    return torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count()))).cuda()


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