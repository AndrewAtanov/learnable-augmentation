from torchvision.datasets import MNIST
from torchvision import transforms
import torch
from torch import nn
from models.mnist_model import FCMNISTVAE, FCMNISTCVAE
import numpy as np


def ohe(labels, n_classes):
    a = np.zeros((len(labels), n_classes))
    a[np.arange(len(labels)), labels] = 1
    return a


def get_dataloaders(data='mnist', train_bs=128, test_bs=500, root='./data', ohe_labels=False):
    if data == 'mnist':
        to_tensor = transforms.ToTensor()
        trainset = MNIST(root, train=True, download=True, transform=to_tensor)
        if ohe_labels:
            x = trainset.train_labels.numpy()
            ohe = np.zeros((len(x), 10))
            ohe[np.arange(ohe.shape[0]), x] = 1
            trainset.train_labels = torch.from_numpy(ohe.astype(np.float32))

        testset = MNIST(root, train=False, download=True, transform=to_tensor)
        if ohe_labels:
            x = testset.test_labels.numpy()
            ohe = np.zeros((len(x), 10))
            ohe[np.arange(ohe.shape[0]), x] = 1
            testset.test_labels = torch.from_numpy(ohe.astype(np.float32))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bs)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs)

    return trainloader, testloader


class FFGKL(nn.Module):
    """KL divergence between standart normal prior and fully-factorize gaussian posterior"""

    def __init__(self):
        super(FFGKL, self).__init__()

    def forward(self, mu, var):
        return 0.5 * (1 + torch.log(var) - mu.pow(2) - var).sum()


class VAEELBOLoss(torch.nn.Module):
    """docstring for ELBOLoss"""

    def __init__(self, likelihood=nn.BCELoss(size_average=False), kl=FFGKL(), use_cuda=False):
        super(VAEELBOLoss, self).__init__()
        self.likelihood = likelihood
        self.kl = kl
        if use_cuda:
            self.likelihood = self.likelihood.cuda()
            self.kl = self.kl.cuda()

    def forward(self, reconstructed, target, *var_params):
        return self.likelihood(reconstructed, target) - self.kl(*var_params)


def get_model(**kwargs):
    model = kwargs['model']
    if model == 'fcmnistvae':
        return FCMNISTVAE(z_dim=kwargs['z_dim'], hidden_dim=kwargs['hidden_dim'], use_cuda=kwargs['cuda'])

    if model == 'fcmnistcvae':
        return FCMNISTCVAE(z_dim=kwargs['z_dim'], hidden_dim=kwargs['hidden_dim'], use_cuda=kwargs['cuda'])
