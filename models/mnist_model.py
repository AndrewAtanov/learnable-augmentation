import torch
from torch import nn
from torch.autograd import Variable


def rnormal(mean, sigma):
    eps = Variable(torch.randn(mean.shape))
    if mean.data.is_cuda:
        eps = eps.cuda()
    return eps * sigma + mean


class Decoder(nn.Module):
    """ Decoder for MNIST model """

    def __init__(self, z_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 784),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.decoder(input)


class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(784, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_sigma = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        x = x.view(-1, 784)

        hidden = self.features(x)

        z_mu = self.fc_mu(hidden)
        z_var = torch.exp(self.fc_sigma(hidden))
        return z_mu, z_var


class VAE(nn.Module):
    """ MNIST VAE """

    def __init__(self, z_dim=20, hidden_dim=400, use_cuda=True):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda

        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)

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
