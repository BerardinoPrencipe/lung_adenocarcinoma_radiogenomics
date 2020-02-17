import torch
import time
import os
import numpy as np
from semseg.loss import dice as dice_loss, tversky, dice_n_classes

eps = 1e-5
use_multi_dice = True
use_tversky = False
alpha, beta = 0.3, 0.7

if use_tversky:
    print('Use Tversky: ', use_tversky)
    print('alpha = ', alpha, ' beta = ', beta)

def get_tversky_loss(outputs, labels):
    outputs = outputs[:, 1, :, :].unsqueeze(dim=1)
    loss = tversky(outputs, labels, alpha=alpha, beta=beta)
    return loss

def get_multi_dice_loss(outputs, labels, device=None):
    labels = labels[:, 0, :, :]
    loss = dice_n_classes(outputs, labels, do_one_hot=True, get_list=False, device=device)
    return loss

def get_loss(outputs, labels, criterion):
    if criterion is None:
        outputs = outputs[:, 1, :, :].unsqueeze(dim=1)
        loss = dice_loss(outputs, labels)
    else:
        labels = labels.squeeze(dim=1)
        loss = criterion(outputs, labels)
    return loss


def train_model(net, optimizer, train_data, config, device=None,
                criterion=None, val_data_list=None, logs_folder=None):
    multi_class =  config['num_outs'] > 2
    print('Multi Class = {}'.format(multi_class))

    print('Start training...')
    # train loop
    for epoch in range(config['epochs']):

        epoch_start_time = time.time()
        running_loss = 0.0

        # lower learning rate
        if epoch == config['low_lr_epoch']:
            for param_group in optimizer.param_groups:
                config['lr'] = config['lr'] / 10
                param_group['lr'] = config['lr']

        # switch to train mode
        net.train()

        for i, data in enumerate(train_data):

            # wrap data in Variables
            inputs, labels = data
            if config['cuda']: inputs, labels = inputs.cuda(), labels.cuda()

            # forward pass and loss calculation
            outputs = net(inputs)

            # get either dice loss or cross-entropy
            if use_multi_dice:
                loss = get_multi_dice_loss(outputs, labels, device=device)
            else:
                loss = get_loss(outputs, labels, criterion)

            # empty gradients, perform backward pass and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # save and print statistics
            running_loss += loss.data

        epoch_end_time = time.time()
        epoch_elapsed_time = epoch_end_time - epoch_start_time
        # print statistics
        if criterion is None:
            print('  [epoch {:04d}] - train dice loss: {:.4f} - time: {:.1f}'
                  .format(epoch + 1, running_loss / (i + 1), epoch_elapsed_time))
        else:
            print('  [epoch {:04d}] - train cross-entropy loss: {:.4f} - time: {:.1f}'
                  .format(epoch + 1, running_loss / (i + 1), epoch_elapsed_time))

        # switch to eval mode
        net.eval()

        all_dice = []
        all_dices = []
        all_accuracy = []

        # only validate every 'val_epochs' epochs
        if epoch % config['val_epochs'] != 0: continue

        if logs_folder is not None:
            checkpoint_path = os.path.join(logs_folder, 'model_epoch_{:04d}.pht'.format(epoch))
            torch.save(net.state_dict(), checkpoint_path)

        # loop through patients
        if val_data_list is None: continue
        eval_start_time = time.time()
        for val_data in val_data_list:

            accuracy = 0.0
            if not multi_class:
                intersect = 0.0
                union = 0.0
            else:
                intersect = np.zeros(config['num_outs'])
                union = np.zeros(config['num_outs'])
            with torch.no_grad():
                for i, data in enumerate(val_data):

                    # wrap data in Variable
                    inputs, labels = data
                    if config['cuda']: inputs, labels = inputs.cuda(), labels.cuda()

                    # inference
                    outputs = net(inputs)

                    # log softmax into softmax
                    if criterion is not None: outputs = outputs.exp()

                    # round outputs to either 0 or 1
                    if not multi_class:
                        outputs = outputs[:, 1, :, :].unsqueeze(dim=1).round()
                    else:
                        outputs = torch.argmax(outputs, dim=1)

                    # accuracy
                    outputs, labels = outputs.cpu().numpy(), labels.cpu().numpy()
                    accuracy += (outputs == labels).sum() / float(outputs.size)

                    if not multi_class:
                        # dice
                        intersect += (outputs + labels == 2).sum()
                        union += np.sum(outputs) + np.sum(labels)
                    else:
                        for cls in range(0, config['num_outs']):
                            outputs_cls = (outputs==cls)
                            labels_cls  = (labels==cls)
                            intersect[cls] += ( np.logical_and(outputs_cls, labels_cls)).sum()
                            union[cls]     += np.sum(outputs_cls) + np.sum(labels_cls)

            all_accuracy.append(accuracy / float(i + 1))
            if not multi_class:
                all_dice.append(1 - (2 * intersect + eps) / (union + eps))

        eval_end_time = time.time()
        eval_elapsed_time = eval_end_time - eval_start_time
        if multi_class:
            print('    Val Accuracy: {:.4f} - Time: {:.1f}'
                  .format(np.mean(all_accuracy), eval_elapsed_time))
            for cls in range(0, config['num_outs']):
                dice_cls = 1 - (2 * intersect[cls] + eps) / (union[cls]  + eps)
                print('      Class [{:02d}] - Dice Loss = {:.4f}'.format(cls, dice_cls))
        else:
            print('    Val Dice Loss: {:.4f} - Val Accuracy: {:.4f} - Time: {:.1f}'
                  .format(np.mean(all_dice), np.mean(all_accuracy), eval_elapsed_time))
    print('Training ended!')
    return net


def get_model_name(model_name):
    ts = time.gmtime()
    ts_hr = time.strftime("%Y-%m-%d__%H_%M_%S", ts)
    final_model_name = model_name + "__" + str(ts_hr) + ".pht"
    return final_model_name