import torch
from semseg.loss import tversky, dice_n_classes, dice as dice_loss, focal_dice_n_classes

eps = 1e-5
LEARNING_RATE_REDUCTION_FACTOR = 10
use_multi_dice = False
use_focal_dice = True
use_vessels_weights = False
use_segments_weights = False
use_tversky = False


#######################
###     DICE LOSS   ###
#######################
def get_loss(outputs, labels, criterion):
    if criterion is None:
        outputs = outputs[:, 1].unsqueeze(dim=1)
        loss = dice_loss(outputs, labels)
    else:
        labels = labels.squeeze(dim=1)
        loss = criterion(outputs, labels)
    return loss


#######################
### MULTI DICE LOSS ###
#######################
print(f'use_multi_dice = {use_multi_dice}')
if use_multi_dice:
    if use_segments_weights:
        weights_balancing_path = 'logs/segments/weights.pt'
        torch_balancing_weights = torch.load(weights_balancing_path)
    elif use_vessels_weights:
        weights_balancing_path = 'logs/vessels_tumors/weights.pt'
        torch_balancing_weights = torch.load(weights_balancing_path)
    else:
        torch_balancing_weights = None

    print('Torch Balancing Weights = {}'.format(torch_balancing_weights))
    gamma = 1.5 # gamma = 2.


def get_multi_dice_loss(outputs, labels, device=None):
    labels = labels[:, 0]
    if use_focal_dice:
        loss = focal_dice_n_classes(outputs, labels, gamma=gamma, weights=torch_balancing_weights,
                                    do_one_hot=True, get_list=False, device=device)
    else:
        loss = dice_n_classes(outputs, labels, do_one_hot=True, get_list=False, device=device)

    return loss


###############
### TVERSKY ###
###############
alpha, beta = 0.3, 0.7
if use_tversky:
    print('Use Tversky: ', use_tversky)
    print('alpha = ', alpha, ' beta = ', beta)


def get_tversky_loss(outputs, labels):
    outputs = outputs[:, 1].unsqueeze(dim=1)
    loss = tversky(outputs, labels, alpha=alpha, beta=beta)
    return loss
