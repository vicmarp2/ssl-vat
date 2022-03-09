import os
import sys
import argparse
import math
from dataloader import get_cifar10, get_cifar100
from test import test_cifar10, test_cifar100
from utils import plot_model, test_error, validation_set

from model.wrn import WideResNet
from train import train

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


# dataloader.py:121: UserWarning UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach()
# or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
import warnings
warnings.filterwarnings("ignore")

def main(args):
    # load data
    if args.dataset == "cifar10":
        args.num_classes = 10
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar10(args,
                                                                       args.datapath)
    if args.dataset == "cifar100":
        args.num_classes = 100
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar100(args,
                                                                        args.datapath)
    
    validation_dataset = validation_set(unlabeled_dataset, args.num_validation, args.num_classes)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    labeled_loader = iter(DataLoader(labeled_dataset,
                                     batch_size=args.train_batch,
                                     shuffle=True,
                                     num_workers=args.num_workers))
    unlabeled_loader = iter(DataLoader(unlabeled_dataset,
                                       batch_size=args.train_batch,
                                       shuffle=True,
                                       num_workers=args.num_workers))

    validation_loader = DataLoader(validation_dataset,
                             batch_size=args.train_batch,
                             shuffle=True,
                             num_workers=args.num_workers)

    test_loader = DataLoader(test_dataset,
                             batch_size=args.test_batch,
                             shuffle=False,
                             num_workers=args.num_workers)

    datasets = {
        'labeled': labeled_dataset,
        'unlabeled': unlabeled_dataset,
        'validation': validation_dataset,
        'test': test_dataset,
    }
    dataloaders = {
        'labeled': labeled_loader,
        'unlabeled': unlabeled_loader,
        'validation': validation_loader,
        'test': test_loader
    }

    args.epoch = math.ceil(args.total_iter / args.iter_per_epoch)

    model = WideResNet(args.model_depth,
                       args.num_classes, widen_factor=args.model_width, dropRate=args.drop_rate)
    model = model.to(device)


    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    # TODO scheduler for learning reate in VAT?
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    criterion = nn.CrossEntropyLoss()

    # train model
    seed = 151
    torch.manual_seed(seed)
    #train(model, datasets, dataloaders, args.modelpath, criterion, optimizer, scheduler, True, True, args)

    # test
    #test_cifar10(test_dataset, './models/best_model_cifar10_4000_0.6_8.pt')
    
    # get test error
    #test_error(test_dataset, './models/best_model_cifar100_10000_0.6_8.pt')

    # %%
    # plot losses
    plot_model('./models/last_model_cifar100_10000_0.6_8.pt')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virtual adversarial training \
                                        of CIFAR10/100 using with pytorch")
    parser.add_argument("--dataset", default="cifar100",
                        type=str, choices=["cifar10", "cifar100"])
    parser.add_argument("--datapath", default="./data/",
                        type=str, help="Path to the CIFAR-10/100 dataset")
    parser.add_argument('--num-labeled', type=int,
                        default=2500, help='Total number of labeled samples')
    parser.add_argument("--lr", default=0.03, type=float,
                        help="The initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="Optimizer momentum")
    # default value was 0.00005. I changed default value, fixmatch paper recomends 0.0005
    parser.add_argument("--wd", default=0.0005, type=float,
                        help="Weight decay")
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--train-batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--test-batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--total-iter', default=1024*100, type=int,
                        help='total number of iterations to run')
    parser.add_argument('--iter-per-epoch', default=1024, type=int,
                        help="Number of iterations to run per epoch")
    parser.add_argument('--num-workers', default=1, type=int,
                        help="Number of workers to launch during training")
    parser.add_argument('--alpha', type=float, default=1, metavar='ALPHA',
                        help='regularization coefficient (default: 0.01)')
    parser.add_argument('--max-grad-norm', default=2, type=float,
                        help='Maximum gradient norm allowed for gradient clipping')
    parser.add_argument("--dataout", type=str, default="./path/to/output/",
                        help="Path to save log files")
    parser.add_argument("--model-depth", type=int, default=28,
                        help="model depth for wide resnet")
    parser.add_argument("--model-width", type=int, default=2,
                        help="model width for wide resnet")
    parser.add_argument("--vat-xi", default=0.6, type=float,
                        help="VAT xi parameter")
    parser.add_argument("--vat-eps", default=5, type=float,
                        help="VAT epsilon parameter")
    parser.add_argument("--vat-iter", default=1, type=int,
                        help="VAT iteration parameter")
    parser.add_argument("--drop-rate", type=int, default=0.0,
                        help="drop out rate for wrn")
    parser.add_argument('--num-validation', type=int,
                        default=1000, help='Total number of validation samples')
    parser.add_argument("--modelpath", default="./models/obs/",
                            type=str, help="Path to the persisted models")
   

    args = parser.parse_args()

    main(args)