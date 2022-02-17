import sys
import argparse
import math
import copy
from os.path import join as pjoin

from dataloader import get_cifar10, get_cifar100
from utils import accuracy, alpha_weight, plot

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import accuracy
from vat import VATLoss


def train (model, datasets, dataloaders, modelpath,
          criterion, optimizer, scheduler, validation, test, args):

    model_subpath = 'cifar10' if args.num_classes == 10 else 'cifar100'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_loss = 1e8
    validation_loss = 1e8
    test_loss = 1e8

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
    # STAGE ONE -> epoch < args.t1
    # alpha for pseudolabeled loss = 0, we just train over the labeled data
    # STAGE TWO -> args.t1 <= epoch <= args.t2
    # alpha gets calculated for weighting the pseudolabeled data
    # we train over labeled and pseudolabeled data
    training_losses = []
    validation_losses = []
    test_losses = []
    for epoch in range(args.epoch):
        running_loss = 0.0
        model.train()
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

            if len(best_model['validation_losses']) == 0 or validation_loss < best_model['validation_losses'][-2]:
                best_model = {
                    'epoch': epoch,
                    'model_state_dict': copy.deepcopy(model.state_dict()),
                    'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                    'training_losses': training_losses,
                    'validation_losses': validation_losses,
                    'test_losses': test_losses,
                    'model_depth' : args.model_depth,
                    'num_classes' : args.num_classes,
                    'model_width' : args.model_width,
                    'drop_rate' : args.drop_rate
                }
                torch.save(best_model, pjoin(modelpath, 'best_model_{}.pt'.format(model_subpath)))
                print('Best model updated with validation loss : {:.5f} '.format(validation_loss))
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
                    acc = accuracy(output_test, y_test)
                    total_accuracy.append(sum(acc))
            test_loss = running_loss / len(test_dataset)
            test_losses.append(test_loss)
            print('Epoch: {} : Test Loss : {:.5f} '.format(
                epoch, test_loss))
            print('Accuracy of the network on test images: %d %%' % (
                sum(total_accuracy)/len(total_accuracy)))

    last_model = {
        'epoch': args.epoch,
        'model_state_dict': copy.deepcopy(model.state_dict()),
        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
        'training_losses': training_losses,
        'validation_losses': validation_losses,
        'test_losses': test_losses,
        'model_depth' : args.model_depth,
        'num_classes' : args.num_classes,
        'model_width' : args.model_width,
        'drop_rate' : args.drop_rate
    }
    torch.save(last_model, pjoin(modelpath, 'last_model_{}.pt'.format(model_subpath)))
    if validation:
        # recover better weights from validation
        model.load_state_dict(best_model['model_state_dict'])
    return model