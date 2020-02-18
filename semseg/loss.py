import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.autograd.function import Function

def dice(outputs, labels):
    eps = 1e-5
    outputs, labels = outputs.float(), labels.float()
    outputs, labels = outputs.flatten(), labels.flatten()
    intersect = torch.dot(outputs, labels)
    union = torch.add(torch.sum(outputs), torch.sum(labels))
    dice_coeff = (2 * intersect + eps) / (union + eps)
    dice_loss = - dice_coeff + 1
    return dice_loss

def tversky(outputs, labels, alpha=0.5, beta=0.5):
    eps = 1e-5
    prob_0 = outputs.flatten()
    prob_1 = 1-prob_0
    gt_0 = labels.bool().flatten()
    gt_1 = torch.bitwise_not(gt_0).flatten()
    tverksy_coeff = (torch.dot(prob_0, gt_0.float()) + eps) \
                     / (torch.dot(prob_0, gt_0.float()) +
                        alpha * torch.dot(prob_0, gt_1.float()) + beta * torch.dot(prob_1, gt_0.float()) + eps)
    tversky_loss = 1 - tverksy_coeff
    return tversky_loss



def one_hot_encode(label, num_classes):
    """

    :param label: Tensor of shape BxHxW
    :param num_classes: K classes
    :return: label_ohe, Tensor of shape BxKxHxW
    """
    label_ohe = torch.zeros((label.shape[0], num_classes, label.shape[1], label.shape[2]))
    for batch_idx, batch_el_label in enumerate(label):
        for cls in range(num_classes):
            label_ohe[batch_idx, cls] = (batch_el_label == cls)
    label_ohe = label_ohe.long()
    return label_ohe

def dice_n_classes(outputs, labels, do_one_hot=False, get_list=False, device=None):
    """
    Computes the Multi-class classification Dice Coefficient.
    It is computed as the average Dice for all classes, each time
    considering a class versus all the others.
    Class 0 (background) is not considered in the average.
    :param outputs: probabilities outputs of the CNN. Shape: [BxKxHxW]
    :param labels:  ground truth                      Shape: [BxKxHxW]
    :param do_one_hot: set to True if ground truth has shape [BxHxW]
    :param get_list:   set to True if you want the list of dices per class instead of average
    :param device: CUDA device on which compute the dice
    :return: Multiclass classification Dice Loss
    """
    num_classes = outputs.shape[1]
    if do_one_hot:
        labels = one_hot_encode(labels, num_classes)
        labels = labels.cuda(device=device)

    dices = list()
    for cls in range(1, num_classes):
        outputs_ = outputs[:, cls, :, :].unsqueeze(dim=1)
        labels_  = labels[:, cls, :, :].unsqueeze(dim=1)
        dice_ = dice(outputs_, labels_)
        dices.append(dice_)
    if get_list:
        return dices
    else:
        return sum(dices) / num_classes

def focal_dice_n_classes(outputs, labels, gamma=2., start_cls=0, weights=None,
                         do_one_hot=False, get_list=False, device=None):
    """
    Computes the Multi-class classification Dice Coefficient.
    It is computed as the average Dice for all classes, each time
    considering a class versus all the others.
    Class 0 (background) is not considered in the average.
    :param outputs: probabilities outputs of the CNN. Shape: [BxKxHxW]
    :param labels:  ground truth                      Shape: [BxKxHxW]
    :param gamma:   gamma coefficient for focal loss
    :param start_cls: set to 0 or to 1, whether you want consider background in average dice or not
    :param do_one_hot: set to True if ground truth has shape [BxHxW]
    :param get_list:   set to True if you want the list of dices per class instead of average
    :param device: CUDA device on which compute the dice
    :return: Multiclass classification Dice Loss
    """

    num_classes = outputs.shape[1]
    if weights is None:
        start_cls = 0
        weights = torch.ones(num_classes)

    weights = weights.cuda(device=device).float()

    if do_one_hot:
        labels = one_hot_encode(labels, num_classes)
        labels = labels.cuda(device=device)

    dices = list()
    for cls in range(start_cls, num_classes):
        outputs_ = outputs[:, cls, :, :].unsqueeze(dim=1)
        labels_  = labels[:, cls, :, :].unsqueeze(dim=1)
        dice_ = dice(outputs_, labels_)
        dice_ = weights[cls] * torch.pow(dice_, gamma)
        dices.append(dice_)
    if get_list:
        return dices
    else:
        return sum(dices) / num_classes