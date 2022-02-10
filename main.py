import argparse
import math
import os
from os.path import join as pjoin
import copy

from dataloader import get_cifar10, get_cifar100
from vat import VATLoss
from utils import accuracy
from model.wrn import WideResNet
from test import test_cifar10

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader


def main(args):
    if args.dataset == "cifar10":
        args.num_classes = 10
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar10(args,
                                                                       args.datapath)
    if args.dataset == "cifar100":
        args.num_classes = 100
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar100(args,
                                                                        args.datapath)
    args.epoch = math.ceil(args.total_iter / args.iter_per_epoch)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    labeled_loader = iter(DataLoader(labeled_dataset,
                                     batch_size=args.train_batch,
                                     shuffle=True,
                                     num_workers=args.num_workers))
    unlabeled_loader = iter(DataLoader(unlabeled_dataset,
                                       batch_size=args.train_batch,
                                       shuffle=True,
                                       num_workers=args.num_workers))
    test_loader = DataLoader(test_dataset,
                             batch_size=args.test_batch,
                             shuffle=False,
                             num_workers=args.num_workers)

    model = WideResNet(args.model_depth,
                       args.num_classes, widen_factor=args.model_width)
    model = model.to(device)

    model_path = "models/obs"

    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    ############################################################################
    # TODO: SUPPLY your code
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    criterion = nn.CrossEntropyLoss()
    ############################################################################

    for epoch in range(args.epoch):
        running_loss = 0
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

            try:
                x_ul, _ = next(unlabeled_loader)
            except StopIteration:
                unlabeled_loader = iter(DataLoader(unlabeled_dataset,
                                                   batch_size=args.train_batch,
                                                   shuffle=True,
                                                   num_workers=args.num_workers))
                x_ul, _ = next(unlabeled_loader)

            x_l, y_l = x_l.to(device), y_l.to(device)
            x_ul = x_ul.to(device)
            ####################################################################
            # TODO: SUPPLY you code
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
            ####################################################################
        scheduler.step()
        epoch_loss = running_loss / (args.iter_per_epoch)
        print('Epoch: {} : Train Loss : {:.5f} '.format(epoch, epoch_loss))

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': copy.deepcopy(model.state_dict()),
        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
        'loss': epoch_loss,
        'model_depth': args.model_depth,
        'num_classes': args.num_classes,
        'model_width': args.model_width
        # 'drop_rate': args.drop_rate
    }
    torch.save(checkpoint, pjoin(model_path, 'model1.pt'))
    test_cifar10(test_loader, model_path)
    # result = test_cifar10(test_loader, model_path)
    # acc = accuracy(result, )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virtual adversarial training \
                                        of CIFAR10/100 using with pytorch")
    parser.add_argument("--dataset", default="cifar10",
                        type=str, choices=["cifar10", "cifar100"])
    parser.add_argument("--datapath", default="./data/",
                        type=str, help="Path to the CIFAR-10/100 dataset")
    parser.add_argument('--num-labeled', type=int,
                        default=4000, help='Total number of labeled samples')
    parser.add_argument("--lr", default=0.03, type=float,
                        help="The initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="Optimizer momentum")
    parser.add_argument("--wd", default=0.00005, type=float,
                        help="Weight decay")
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--train-batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--test-batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--total-iter', default=1024 * 512, type=int,
                        help='total number of iterations to run')
    parser.add_argument('--iter-per-epoch', default=1024, type=int,
                        help="Number of iterations to run per epoch")
    parser.add_argument('--num-workers', default=1, type=int,
                        help="Number of workers to launch during training")
    parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                        help='regularization coefficient (default: 0.01)')
    parser.add_argument("--dataout", type=str, default="./path/to/output/",
                        help="Path to save log files")
    parser.add_argument("--model-depth", type=int, default=28,
                        help="model depth for wide resnet")
    parser.add_argument("--model-width", type=int, default=2,
                        help="model width for wide resnet")
    parser.add_argument("--vat-xi", default=10.0, type=float,
                        help="VAT xi parameter")
    parser.add_argument("--vat-eps", default=1.0, type=float,
                        help="VAT epsilon parameter")
    parser.add_argument("--vat-iter", default=1, type=int,
                        help="VAT iteration parameter")
    # Add more arguments if you need them
    # Describe them in help
    # You can (and should) change the default values of the arguments

    args = parser.parse_args()

    main(args)