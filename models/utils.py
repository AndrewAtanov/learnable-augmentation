import torch

print('aAA')

def rnormal(mean, sigma, batch_size):
    eps = torch.randn((batch_size, ) + mean.shape)
    return eps * sigma.view((1,) + sigma.shape) + mean.view((1,) + mean.shape)
