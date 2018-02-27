import torch
from torch import nn
from .utils import rnormal


class VAE(nn.Module):
    """ VAE """

    def __init__(self, encoder, decoder, use_cuda=True):
        super(VAE, self).__init__()
        self.use_cuda = use_cuda

        self.encoder = encoder
        self.decoder = decoder

        if use_cuda:
            self.cuda()

    def encode(self, input):
        return self.encoder(input)

    def decode(self, input):
        return self.decoder(input)

    def forward(self, input):
        mu, var = self.encoder(input)
        z = rnormal(mu, torch.sqrt(var))

        return mu, var, self.decoder(z)


class CVAE(VAE):
    """ VAE """

    def __init__(self, encoder, decoder, use_cuda=True):
        super(CVAE, self).__init__(encoder, decoder, use_cuda=use_cuda)
        self.use_cuda = use_cuda
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, labels):
        mu, var = self.encode(torch.cat([input, labels], dim=1))
        z = rnormal(mu, torch.sqrt(var))

        return mu, var, self.decoder(torch.cat([z, labels], dim=1))
