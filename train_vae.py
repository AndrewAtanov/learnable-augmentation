import argparse
import numpy as np
import torch
from torch.autograd import Variable
import utils


def train(trainloader, testloader, vae, optimizer, criterion, args):
    for epoch in range(args.num_epochs):
        train_loss = 0.

        for i, (batch, y) in enumerate(trainloader):
            optimizer.zero_grad()

            if args.cuda:
                batch = batch.cuda()
                y = y.cuda()

            batch, y = Variable(batch), Variable(y)
            mu, sigma, reconstructed = vae(batch, y) if 'cvae' in args.model else vae(batch)

            loss = criterion(reconstructed, batch.view(-1, 784), mu, sigma)

            loss.backward()
            optimizer.step()

            train_loss += loss.data.cpu()[0]

        if args.verbose > 0:
            print('{} epoch | train ELBO {:.4f} '.format(epoch + 1, -train_loss / len(trainloader.dataset)))

            if epoch % args.test_frequency == 0:
                test_loss = 0.
                for i, (batch, y) in enumerate(trainloader):
                    if args.cuda:
                        batch = batch.cuda()
                        y = y.cuda()

                    batch, y = Variable(batch), Variable(y)
                    mu, sigma, reconstructed = vae(batch, y) if 'cvae' in args.model else vae(batch)

                    loss = criterion(reconstructed, batch.view(-1, 784), mu, sigma)
                    test_loss += loss.data.cpu()[0]

                print('{} epoch validation | test ELBO {:.4f} '.format(epoch + 1, -test_loss / len(testloader.dataset)))

        torch.save(vae.state_dict(), args.log_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='mnist')
    parser.add_argument('--model', default='fcmnistvae')
    parser.add_argument('--log_file', default='vae_params')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--test_frequency', default=10, type=int)
    parser.add_argument('--num_epochs', default=15, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--z_dim', default=20, type=int)
    parser.add_argument('--hidden_dim', default=400, type=int, help='FC size for FC MNIST')
    parser.add_argument('--verbose', default=1, type=int)
    args = parser.parse_args()

    args.cuda = torch.cuda.device_count() != 0

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    vae = utils.get_model(**vars(args))
    optimizer = torch.optim.Adam(vae.parameters())
    criterion = utils.VAEELBOLoss(use_cuda=args.cuda)

    trainloader, testloader = utils.get_dataloaders(data=args.data, train_bs=args.batch_size, ohe_labels=True)

    train(trainloader, testloader, vae, optimizer, criterion, args)
