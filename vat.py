import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision

def L2Norm(r):
    r_reshaped = r.view(r.shape[0], -1, *(1 for _ in range(r.dim() - 2)))
    r /= torch.norm(r_reshaped, dim=1, keepdim=True) + 1e-10
    return r

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VATLoss(nn.Module):

    def __init__(self, args):
        super(VATLoss, self).__init__()
        self.xi = args.vat_xi
        self.eps = args.vat_eps
        self.vat_iter = args.vat_iter

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)

        # prepare random unit tensor from gaussian distribution
        r = torch.randn(x.shape).sub(0.5).to(x.device)
        r = L2Norm(r)

        model.eval()
        for _ in range(self.vat_iter):
            r.requires_grad_()
            advExamples = x + self.xi * r
            advPredictions = F.log_softmax(model(advExamples), dim=1)
            adv_distance = F.kl_div(advPredictions, pred, reduction='batchmean')
            adv_distance.backward()
            r = L2Norm(r.grad)
            model.zero_grad()
        model.train()

        # calc loss
        r_adv = r * self.eps
        advImage = x + r_adv
        advExamples = model(advImage)
        advPredictions = F.log_softmax(advExamples, dim=1)
        loss = F.kl_div(advPredictions, pred, reduction='batchmean')

        '''writer = SummaryWriter()
        grid_x = torchvision.utils.make_grid(x)
        writer.add_image('image', grid_x, 0)
        grid = torchvision.utils.make_grid(advImage)
        writer.add_image('perturbation', grid, 0 + 1)'''

        return loss