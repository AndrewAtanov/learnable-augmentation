from torchvision.datasets import MNIST
from torchvision import transforms
import torch
from torch import nn


def get_dataloaders(data='mnist', train_bs=128, test_bs=500, root='./data'):
    if data == 'mnist':
        to_tensor = transforms.ToTensor()
        trainset = MNIST(root, train=True, download=True, transform=to_tensor)
        testset = MNIST(root, train=False, download=True, transform=to_tensor)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bs)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs)

    return trainloader, testloader


class ELBOLoss(torch.nn.Module):
    """docstring for ELBOLoss"""

    def __init__(self, use_cuda=False):
        super(ELBOLoss, self).__init__()
        self.bce = nn.BCELoss(size_average=False)
        if use_cuda:
            self.bce = self.bce.cuda()

    def forward(self, reconstructed, target, mu, var):
        bce = self.bce(reconstructed, target)
        kl = -0.5 * (1 + torch.log(var) - mu.pow(2) - var).sum()

        return bce + kl
