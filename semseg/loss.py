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
''' 
gt = torch.ones([1,16,16])
pred = torch.ones([1,16,16]) * 0.95

d = dice(pred, gt)
t = tversky(pred, gt)
print('Dice    Loss = {}'.format(d))
print('Tversky Loss = {}'.format(t))
'''

class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
        :param predict: (n, c, h, w)
        :param target:  (n, h, w)
        :param weight:  (Tensor, optional): a manual rescaling weight given to each class.
                                            If given, has to be a Tensor of size "nclasses"
        :return:
        """

        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{} vs {}".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{} vs {}".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{} vs {}".format(predict.size(3), target.size(2))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1,2).transpose(2,3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss