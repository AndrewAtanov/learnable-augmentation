from torchvision.datasets import MNIST
import os
from torchvision import transforms
import torch
from torch import nn
from models.mnist_model import FCMNISTVAE, FCMNISTCVAE
import numpy as np
from sklearn.cross_validation import train_test_split


class AccCounter:
    """
    Class for count accuracy during pass through data with mini-batches.
    """
    def __init__(self):
        self.__n_objects = 0
        self.__sum = 0

    def add(self, outputs, targets):
        """
        Compute and save stats needed for overall accuracy.
        :param outputs: ndarray of predicted values (logits or probabilities)
        :param targets: ndarray of labels with the same length as first dimension of _outputs_
        """
        self.__sum += np.sum(outputs.argmax(axis=1) == targets)
        self.__n_objects += outputs.shape[0]

    def acc(self):
        """
        Compute current accuracy.
        :return: float accuracy.
        """
        return self.__sum * 1. / self.__n_objects

    def flush(self):
        """
        Flush stats.
        :return:
        """
        self.__n_objects = 0
        self.__sum = 0


def ohe(labels, n_classes):
    a = np.zeros((len(labels), n_classes))
    a[np.arange(len(labels)), labels] = 1
    return a


def get_dataloaders(data='mnist', train_bs=128, test_bs=500, root='./data', ohe_labels=False, train_fraction=1.):
    to_tensor = transforms.ToTensor()
    if data == 'mnist':
        trainset = MNIST(root, train=True, download=True, transform=to_tensor)
        if train_fraction < 1.:
            data, _, labels, _ = train_test_split(trainset.train_data.numpy(), trainset.train_labels.numpy(),
                                                  stratify=trainset.train_labels.numpy(), train_size=train_fraction)
            trainset.train_data, trainset.train_labels = torch.ByteTensor(data), torch.LongTensor(labels)

        idx = torch.LongTensor(np.where(trainset.train_labels.numpy() == 0)[0])
        trainset.train_data = trainset.train_data[idx]
        trainset.train_labels = trainset.train_labels[idx]

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
    elif data == 'not-mnist':
        trainset = MNIST(root=os.path.join(root, 'not-mnist'), train=False,
                         download=True, transform=to_tensor)
        testset = MNIST(root=os.path.join(root, 'not-mnist'), train=False,
                        download=True, transform=to_tensor)
    else:
        raise NotImplementedError

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
