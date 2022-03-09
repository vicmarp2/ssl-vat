import sys
import argparse
import math
import copy
import os
from os.path import join as pjoin

from dataloader import get_cifar10, get_cifar100
from utils import accuracy, plot_model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import accuracy
from vat import VATLoss
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.squeeze(0).cuda().detach().cpu()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(npimg.T)

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(32, 32))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        #ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
        #    classes[preds[idx]],
        #    probs[idx] * 100.0,
        #    classes[labels[idx]]),
        #            color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

# default `log_dir` is "runs" - we'll be more specific here
#writer = SummaryWriter('runs/cifar10')

def train (model, datasets, dataloaders, modelpath,
          criterion, optimizer, scheduler, validation, test, args):

    if not os.path.isdir(modelpath):
        os.makedirs(modelpath)
    model_subpath = 'cifar10' if args.num_classes == 10 else 'cifar100'
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if validation:
        best_model = {
            'epoch': 0,
            'model_state_dict': copy.deepcopy(model.state_dict()),
            'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
            'training_losses': [],
            'validation_losses': [],
            'test_losses': [],
            'model_depth' : args.model_depth,
            'num_classes' : args.num_classes,
            'model_width' : args.model_width,
            'drop_rate' : args.drop_rate,
        } 
    # access datasets and dataloders
    labeled_dataset = datasets['labeled']
    labeled_loader = dataloaders['labeled']
    unlabeled_loader = dataloaders['unlabeled']
    unlabeled_dataset = datasets['unlabeled']
    if validation:
        validation_dataset = datasets['validation']
        validation_loader = dataloaders['validation']
    if test:
        test_dataset = datasets['test']
        test_loader = dataloaders['test']

    print('Training started')
    print('-' * 20)
    model.train()
    # train

    training_losses = []
    validation_losses = []
    test_losses = []
    for epoch in range(args.epoch):
        running_loss = 0.0
        for i in range(args.iter_per_epoch):
            try:
                x_l, y_l = next(labeled_loader)
            except StopIteration:
                labeled_loader = iter(DataLoader(labeled_dataset,
                                                 batch_size=args.train_batch,
                                                 shuffle=True,
                                                 num_workers=args.num_workers))
                x_l, y_l = next(labeled_loader)
            x_l, y_l = x_l.to(device), y_l.to(device)
            
        
            try:
                x_ul, y_ul = next(unlabeled_loader)
            except StopIteration:
                unlabeled_loader = iter(DataLoader(unlabeled_dataset,
                                                batch_size=args.train_batch,
                                                shuffle=True,
                                                num_workers=args.num_workers))
                x_ul, _ = next(unlabeled_loader)
            x_ul = x_ul.to(device)
    
         
            optimizer.zero_grad()
            # vat_loss = VATLoss(args.vat_xi, args.vat_eps, args.vat_iter)
            vat_loss = VATLoss(args)
            lds = vat_loss(model, x_ul)
            output = model(x_l)
            classification_loss = criterion(output, y_l)
            loss = classification_loss + args.alpha * lds
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()

            running_loss += loss.item()
        training_loss = running_loss/(args.iter_per_epoch)
        training_losses.append(training_loss)
        print('Epoch: {} : Train Loss : {:.5f} '.format(
            epoch, training_loss))
        
        # Calculate loss for validation set every epoch
        # Save the best model
        # TODO implement early stopping?
        running_loss = 0.0
        if validation:
            model.eval()
            for x_val, y_val in validation_loader:
                with torch.no_grad():
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    output_val = model(x_val)
                    loss = criterion(output_val, y_val)

                    running_loss += loss.item() * x_val.size(0)

            validation_loss = running_loss / len(validation_dataset)
            validation_losses.append(validation_loss)
            print('Epoch: {} : Validation Loss : {:.5f} '.format(
            epoch, validation_loss))

            if best_model['epoch'] == 0 or validation_loss < best_model['validation_losses'][-1]:
                best_model = {
                    'epoch': epoch,
                    'model_state_dict': copy.deepcopy(model.state_dict()),
                    'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                    'training_losses':  copy.deepcopy(training_losses),
                    'validation_losses': copy.deepcopy(validation_losses),
                    'test_losses': copy.deepcopy(test_losses),
                    'model_depth' : args.model_depth,
                    'num_classes' : args.num_classes,
                    'model_width' : args.model_width,
                    'drop_rate' : args.drop_rate
                }
                torch.save(best_model, pjoin(modelpath, 'best_model_{}_{}_{}_{}.pt'.format(model_subpath, args.num_labeled, args.vat_xi, args.vat_eps)))
                print('Best model updated with validation loss : {:.5f} '.format(validation_loss))
            model.train()
        # update learning rate
        scheduler.step()

        # Check test error with current model over test dataset
        running_loss = 0.0
        if test:
            total_accuracy = []
            test_loss = 0.0
            model.eval()
            for x_test, y_test in test_loader:
                with torch.no_grad():
                    x_test, y_test = x_test.to(device), y_test.to(device)
                    output_test = model(x_test)                              
                    loss = criterion(output_test, y_test)
                    running_loss += loss.item() * x_test.size(0)
                    ''' # ...log the running loss
                    writer.add_scalar('training loss',
                                      running_loss / 1000,
                                      epoch * len(test_loader) + i)

                    # ...log a Matplotlib Figure showing the model's predictions on a
                    # random mini-batch
                    writer.add_figure('predictions vs. actuals',
                                      plot_classes_preds(model, x_test, y_test),
                                      global_step=epoch * len(test_loader) + i)'''
                    acc = accuracy(output_test, y_test)
                    total_accuracy.append(sum(acc))
            test_loss = running_loss / len(test_dataset)
            test_losses.append(test_loss)
            print('Epoch: {} : Test Loss : {:.5f} '.format(
                epoch, test_loss))
            print('Accuracy of the network on test images: %d %%' % (
                sum(total_accuracy)/len(total_accuracy)))

    last_model = {
        'epoch': epoch,
        'model_state_dict': copy.deepcopy(model.state_dict()),
        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
        'training_losses':  copy.deepcopy(training_losses),
        'validation_losses': copy.deepcopy(validation_losses),
        'test_losses': copy.deepcopy(test_losses),
        'model_depth' : args.model_depth,
        'num_classes' : args.num_classes,
        'model_width' : args.model_width,
        'drop_rate' : args.drop_rate
    }
    torch.save(best_model, pjoin(modelpath, 'last_model_{}_{}_{}_{}.pt'.format(model_subpath, args.num_labeled, args.vat_xi, args.vat_eps)))
    if validation:
        # recover better weights from validation
        model.load_state_dict(best_model['model_state_dict'])
    return model