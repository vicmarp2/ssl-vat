import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
        advImage = x + r_adv
        advExamples = model(advImage)
        advPredictions = F.softmax(advExamples, dim=1)
        loss = F.kl_div(advPredictions, pred)

        #figure, ax = plt.subplots(1, 2, figsize=(32, 32))
        #ax[0].imshow(x.squeeze(0).cuda().detach().cpu())
        #ax[0].set_title('Clean Example', fontsize=20)
        # ax[1].imshow(r_adv)
        # ax[1].set_title('Perturbation', fontsize=20)
        #ax[1].imshow(advImage.squeeze(0).cuda().detach().cpu())
        #ax[1].set_title('Adversarial Example', fontsize=20)
        #plt.show()

        return loss