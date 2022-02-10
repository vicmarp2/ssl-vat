

import torch
import torch.nn as nn
import torch.nn.functional as F

def L2Norm(r):
    r_reshaped = r.view(r.shape[0], -1, *(1 for _ in range(r.dim() - 2)))
    r /= torch.norm(r_reshaped, dim=1, keepdim=True) + 1e-8
    return r

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

        for _ in range(self.vat_iter):
            r.requires_grad_()
            advExamples = model(x + self.xi * r)
            advPredictions = F.softmax(advExamples, dim=1)
            adv_distance = F.kl_div(advPredictions, pred)
            adv_distance.backward()
            r = L2Norm(r.grad)
            model.zero_grad()

        # calc loss
        r_adv = r * self.eps
        advExamples = model(x + r_adv)
        advPredictions = F.softmax(advExamples, dim=1)
        loss = F.kl_div(advPredictions, pred)

        return loss