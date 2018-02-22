import torch
from torch import nn
from .vae import VAE, CVAE


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
    def __init__(self, z_dim, hidden_dim, input_dim=784):
        super(Encoder, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_sigma = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        hidden = self.features(x)

        z_mu = self.fc_mu(hidden)
        z_var = torch.exp(self.fc_sigma(hidden))
        return z_mu, z_var


class FCMNISTVAE(VAE):
    """docstring for MNISTVAE"""

    def __init__(self, z_dim=20, hidden_dim=400, use_cuda=True):
        super(FCMNISTVAE, self).__init__(encoder=Encoder(z_dim, hidden_dim), decoder=Decoder(z_dim, hidden_dim),
                                         use_cuda=use_cuda)

    def forward(self, input):
        input = input.view((-1, 784))
        return super().forward(input)


class FCMNISTCVAE(CVAE):
    """docstring for MNISTVAE"""

    def __init__(self, z_dim=20, hidden_dim=400, use_cuda=True):
        super(FCMNISTCVAE, self).__init__(encoder=Encoder(z_dim, hidden_dim, input_dim=784 + 10),
                                          decoder=Decoder(z_dim + 10, hidden_dim),
                                          use_cuda=use_cuda)

    def forward(self, input, labels):
        input = input.view((-1, 784))

        return super().forward(input, labels)
