import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.01):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def pairwise_distances(x):
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()

def GaussianKernelMatrix(x, alpha, ell):
    pairwise_distances_ = pairwise_distances(x)
    r = torch.sqrt(pairwise_distances_)
    kernel_matrix = alpha * (1 + torch.sqrt(3) * r / ell) * torch.exp(-torch.sqrt(3) * r / ell)
    return kernel_matrix


def HSIC(x, y, s_x=1, s_y=1):
    m, _ = x.shape  # batch size
    K = GaussianKernelMatrix(x, s_x, 1)
    L = GaussianKernelMatrix(y, s_y, 1)
    H = torch.eye(m) - 1.0 / m * torch.ones((m, m))
    H = H.float().cuda()
    HSIC = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1) ** 2)
    return HSIC


def loss_func(z, z_prior, y, num_cls):

    y_valid = [i_cls in y for i_cls in range(num_cls)]
    z_mean = torch.stack([z[y == i_cls].mean(dim=0) for i_cls in range(num_cls)], dim=0)  # 60 256
    hsic = HSIC(z_mean[y_valid], z_prior[y_valid].to(z.device))
    return hsic


