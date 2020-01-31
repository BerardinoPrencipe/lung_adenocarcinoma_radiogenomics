import torch
import time
import os
import numpy as np
from semseg.loss import dice as dice_loss, tversky

use_tversky = False
alpha, beta = 0.3, 0.7

print('Use Tversky: ', use_tversky)
print('alpha = ', alpha, ' beta = ', beta)

def get_tversky_loss(outputs, labels):
    outputs = outputs[:, 1, :, :].unsqueeze(dim=1)
    loss = tversky(outputs, labels, alpha=alpha, beta=beta)
    return loss

def get_loss(outputs, labels, criterion):
    if criterion is None:
        outputs = outputs[:, 1, :, :].unsqueeze(dim=1)
        loss = dice_loss(outputs, labels)
    else:
        labels = labels.squeeze(dim=1)
        loss = criterion(outputs, labels)
    return loss


def train_model(net, optimizer, train_data, config,
                criterion=None, val_data_list=None, logs_folder=None):
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
            intersect = 0.0
            union = 0.0

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
                    outputs = outputs[:, 1, :, :].unsqueeze(dim=1).round()

                    # accuracy
                    outputs, labels = outputs.data.cpu().numpy(), labels.data.cpu().numpy()
                    accuracy += (outputs == labels).sum() / float(outputs.size)

                    # dice
                    intersect += (outputs + labels == 2).sum()
                    union += np.sum(outputs) + np.sum(labels)

            all_accuracy.append(accuracy / float(i + 1))
            all_dice.append(1 - (2 * intersect + 1e-5) / (union + 1e-5))
        eval_end_time = time.time()
        eval_elapsed_time = eval_end_time - eval_start_time
        print('    val dice loss: {:.4f} - val accuracy: {:.4f} - time: {:.1f}'
              .format(np.mean(all_dice), np.mean(all_accuracy), eval_elapsed_time))
    print('Training ended!')
    return net


def get_model_name(model_name):
    ts = time.gmtime()
    ts_hr = time.strftime("%Y-%m-%d__%H_%M_%S", ts)
    final_model_name = model_name + "__" + str(ts_hr) + ".pht"
    return final_model_name