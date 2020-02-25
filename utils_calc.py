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