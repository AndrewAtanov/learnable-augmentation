import argparse
import numpy as np
import torch
from torch.autograd import Variable
from models.mnist_model import VAE
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--data', default='mnist')
parser.add_argument('--cuda', dest='cuda', action='store_true')
parser.add_argument('--no-cuda', dest='cuda', action='store_false')
parser.set_defaults(cuda=True)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--test_frequency', default=10, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--z_dim', default=20, type=int)

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

vae = VAE(z_dim=args.z_dim, use_cuda=args.cuda)
optimizer = torch.optim.Adam(vae.parameters())
criterion = utils.ELBOLoss(use_cuda=args.cuda)

trainloader, testloader = utils.get_dataloaders(data=args.data, train_bs=args.batch_size)

for epoch in range(args.num_epochs):
    train_loss = 0.

    for i, (batch, _) in enumerate(trainloader):
        optimizer.zero_grad()

        if args.cuda:
            batch = batch.cuda()

        batch = Variable(batch)
        mu, sigma, reconstructed = vae(batch)

        loss = criterion(reconstructed, batch.view(-1, 784), mu, sigma)

        loss.backward()
        optimizer.step()

        train_loss += loss.data.cpu()[0]

    print('{} epoch | train ELBO {:.4f} '.format(epoch + 1, -train_loss / len(trainloader.dataset)))

    if epoch % args.test_frequency == 0:
        test_loss = 0.
        for i, (batch, _) in enumerate(trainloader):
            if args.cuda:
                batch = batch.cuda()

            batch = Variable(batch)
            mu, sigma, reconstructed = vae(batch)

            loss = criterion(reconstructed, batch.view(-1, 784), mu, sigma)
            test_loss += loss.data.cpu()[0]

        print('{} epoch validation | test ELBO {:.4f} '.format(epoch + 1, -test_loss / len(testloader.dataset)))

    torch.save(vae.state_dict(), 'vae_params')
