import torch
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
    dice_loss = 1 - dice_coeff
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

# Debug only
""" 
gt = torch.ones([5,16,16])
pred = torch.ones([5,16,16]) * 0.95

d = dice(pred, gt)
t = tversky(pred, gt)
"""