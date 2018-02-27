import torch
from torch import nn
import numpy as np
from models.utils import rnormal
from torch.autograd import Variable
from train_vae import train
import utils


class VAESampler(nn.Module):
    """docstring for VAESampler"""

    def __init__(self, cvae, n_classes):
        super(VAESampler, self).__init__()
        self.cvae = cvae
        self.n_classes = n_classes

    def train(self, trainloader, testloader, args):
        if args.verbose > 0:
            print('==> Train CVAE data augmentator...')

        self.optimizer = torch.optim.Adam(self.cvae.parameters())
        self.criterion = utils.VAEELBOLoss(use_cuda=args.cuda)

        train(trainloader, testloader, self.cvae, self.optimizer, self.criterion, args)

        if args.verbose > 0:
            print('==> Done training CVAE augmentator.')

    def sample(self, bs):
        y = np.random.choice(self.n_classes, size=(bs,))
        labels = Variable(torch.FloatTensor(utils.ohe(y, self.n_classes)))
        y = torch.LongTensor(y)

        mu = Variable(torch.zeros((bs, self.cvae.encoder.z_dim)))
        var = Variable(torch.ones((bs, self.cvae.encoder.z_dim, )))
        if self.cvae.use_cuda:
            mu, var = mu.cuda(), var.cuda()
            labels, y = labels.cuda(), y.cuda()
        z = rnormal(mu, torch.sqrt(var))

        batch = self.cvae.decode(torch.cat([z, labels], dim=1))

        return batch.data.cpu().view((bs, 1, 28, 28)), y.cpu()

    def mix_sampler(self, dataloader, sample_bs):
        for x, y in dataloader:
            sample_x, sample_y = self.sample(sample_bs)
            batch_x = torch.cat((x, sample_x))
            batch_y = torch.cat((y, sample_y))
            yield batch_x, batch_y
