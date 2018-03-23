import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.lin_features = nn.Sequential(
            nn.Linear(200, 400),
        )

        # 20 * 20
        self.conv_features = nn.Sequential(
            nn.ConvTranspose2d(1, 1, 3),
            nn.ELU(),
            nn.ConvTranspose2d(1, 1, 3),
            nn.ELU(),
            nn.ConvTranspose2d(1, 1, 5),
            nn.Sigmoid()
        )

    def forward(self, input):
        out = self.lin_features(input)
        out = self.conv_features(out.view(out.size(0), 1, 20, 20))
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_features = nn.Sequential(
            # input feat, output feat, kernel
            # 28 - 4*3 = 16
            nn.Conv2d(1, 4, 5),
            nn.ELU(),
            nn.Conv2d(4, 16, 5),
            nn.ELU(),
            nn.Conv2d(16, 32, 5),
            nn.ELU(),
        )
        self.lin_features = nn.Sequential(
            nn.Linear(32*16*16, 1000),
            nn.ELU(),
            nn.Linear(1000, 500),
            nn.ELU(),
            nn.Linear(500, 100),
            nn.ELU(),
        )

    def forward(self, x):
        out = self.conv_features(x.view(x.size(0), 1, 28, 28))
        out = self.lin_features(out.view(-1, 32*16*16))
        return out


class AEGenerator(nn.Module):
    def __init__(self, encoder, decoder, use_cuda=True):
        super(AEGenerator, self).__init__()
        self.use_cuda = use_cuda

        self.encoder = encoder
        self.decoder = decoder
        self.z_fc = nn.Linear(500, 100)

        if use_cuda:
            self.cuda()

    def forward(self, conditional_input):
        bs = conditional_input.size(0)
        r = self.encoder(conditional_input)
        z_inputs = Variable(torch.randn(bs, 500).cuda())
        z = self.z_fc(z_inputs)
        return self.decoder(torch.cat((r, z), dim=1))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(784*2, 2)
        #self.fc2 = nn.Linear(hidden_size, hidden_size)
        #self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x1, x2):
        input = torch.cat((x1.view(x1.size(0), -1), x2.view(x2.size(0), -1)), dim=1)
        #x = F.elu(self.fc1(input))
        #x = F.elu(self.fc2(x))
        x = input
        return F.log_softmax(self.fc1(x))


class DAGAN(nn.Module):
    def __init__(self, generator, discriminator, use_cuda=True):
        super(DAGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        if use_cuda:
            self.cuda()

    def compute_loss_d(self, input_a, input_b):
        x_g = self.generator(input_a)

        d_fake = self.discriminator(x_g.detach(), input_a)
        d_real  = self.discriminator(input_b, input_a)

        d_loss = - (torch.mean(d_real[:, 0]) + torch.mean(d_fake[:, 1]))
        return d_loss

    def compute_loss_g(self, input_a):
        x_g = self.generator(input_a)
        d_fake = self.discriminator(x_g, input_a)
        g_loss = torch.mean(d_fake[:, 1])
        return g_loss

