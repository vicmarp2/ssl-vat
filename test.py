import torch
import os
from os.path import join as pjoin
from dataloader import get_cifar10, get_cifar100
from model.wrn import WideResNet
from utils import accuracy
import torch.nn.functional as F


def test_cifar10(testdataset, filepath="./path/to/model.pth.tar"):
    '''
    args:
        testdataset : (torch.utils.data.Dataset)
        filepath    : (str) The path to the model file that is saved
    returns : (torch.Tensor) logits of the testdataset with shape
                [num_samples, 10]. Apply softmax to the logits

    Description:
        This function loads the model given in the filepath and returns the
        logits of the testdataset which is a torch.utils.data.Dataset. You can
        save the arguments needed to load the models (e.g. width, depth etc)
        with the model file. Assume testdataset is like CIFAR-10. Test this
        function with the testdataset returned by get_cifar10()
    '''
    # TODO: SUPPLY the code for this function
    cp = torch.load(pjoin(filepath, 'model1.pt'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WideResNet(cp['model_depth'],
                       cp['num_classes'], widen_factor=cp['model_width'])
    model = model.to(device)
    model.load_state_dict(cp['model_state_dict'])
    correct = 0
    acct = []
    output = torch.Tensor([])
    with torch.no_grad():
        for data in testdataset:
            x, y = data
            x, y = x.to(device), y.to(device)
            out = model(x)
            s_out = torch.zeros_like(out)
            s_out = F.softmax(out, dim=1)
            output = torch.stack(tuple(s_out), 0)
            acc = accuracy(s_out, y)
            acct.append(sum(acc))
        print('Accuracy of the network on test images: %d %%' % (
                sum(acct) / len(acct)))
    return output


def test_cifar100(testdataset, filepath="./path/to/model.pth.tar"):
    '''
    args:
        testdataset : (torch.utils.data.Dataset)
        filepath    : (str) The path to the model file that is saved
    returns : (torch.Tensor) logits of the testdataset with shape
                [num_samples, 100]. Apply softmax to the logits

    Description:
        This function loads the model given in the filepath and returns the 
        logits of the testdataset which is a torch.utils.data.Dataset. You can
        save the arguments needed to load the models (e.g. width, depth etc) 
        with the model file. Assume testdataset is like CIFAR-100. Test this
        function with the testdataset returned by get_cifar100()
    '''
    # TODO: SUPPLY the code for this function
    raise NotImplementedError