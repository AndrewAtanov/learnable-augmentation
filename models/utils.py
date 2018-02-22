import torch
from torch.autograd import Variable


def rnormal(mean, sigma):
    eps = Variable(torch.randn(mean.shape))
    if mean.data.is_cuda:
        eps = eps.cuda()
    return eps * sigma + mean
