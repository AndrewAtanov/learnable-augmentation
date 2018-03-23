import argparse
import numpy as np
import torch
from torch.autograd import Variable
from models.dagan_model import DAGAN, Encoder, Decoder, AEGenerator, Discriminator
import utils


def train(trainloader, testloader, dagan, opt_g, opt_d, args):
    g_steps = 200
    d_steps = 10

    accs = []
    train_g_loss_hist = []
    train_d_loss_hist = []
    for epoch in range(args.num_epochs):

        train_d_loss = 0
        steps = 0
        for i, (batch, y) in enumerate(trainloader):
            if i > d_steps:
                break

            half = batch.shape[0]//2
            input_a, input_b = Variable(batch[:half]).cuda(), Variable(batch[half:2*half]).cuda()

            x_g = dagan.generator(input_a)

            d_fake = dagan.discriminator(x_g.detach(), input_a)
            d_real  = dagan.discriminator(input_b, input_a)
            acc_fake = (d_fake[:, 1].data.cpu().numpy() > np.log(0.5)).mean()
            acc_real = (d_real[:, 0].data.cpu().numpy() > np.log(0.5)).mean()
            acc = (acc_fake + acc_real) / 2
            accs.append(acc)

            d_loss = - (torch.mean(d_real[:, 0]) + torch.mean(d_fake[:, 1]))
            train_d_loss += d_loss.data.cpu()[0]
            steps += 1
            train_d_loss_hist.append(d_loss.data.cpu()[0])

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # is disc gets too good
            if acc > 0.8:
                break

            train_d_loss /= steps

        train_g_loss = 0
        steps = 0
        for i, (batch, y) in enumerate(trainloader):
            if i > g_steps:
                break

            x_g = dagan.generator(input_a)

            d_fake = dagan.discriminator(x_g, input_a)
            g_loss = torch.mean(d_fake[:, 1])
            acc = (d_fake[:, 1].data.cpu().numpy() < np.log(0.5)).mean()
            accs.append(acc)
            train_g_loss += g_loss.data.cpu()[0]
            steps += 1
            train_g_loss_hist.append(g_loss.data.cpu()[0])

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()
        train_g_loss /= steps

        if args.verbose > 0:
            print('{} epoch | train g_loss {:.4f} d_loss {:.4f}'.format(epoch + 1, train_g_loss, train_d_loss))

        torch.save(dagan.state_dict(), args.log_file)
        np.save('g_hist', train_g_loss_hist)
        np.save('d_hist', train_d_loss_hist)
        np.save('acc', accs)


        # if epoch % args.test_frequency == 0:
        #     test_g_loss = 0
        #     test_d_loss = 0
        #     for i, (batch, y) in enumerate(testloader):
        #         if args.cuda:
        #             batch = batch.cuda()
        #             y = y.cuda()

        #         half = batch.shape[0]//2
        #         input_a, input_b = Variable(batch[:half]), Variable(batch[half:2*half])
        #         d_loss = dagan.compute_loss_d(input_a, input_b)
        #         test_d_loss += d_loss.data[0]
        #         g_loss = dagan.compute_loss_g(input_a)
        #         test_g_loss += g_loss.data.cpu()[0]

        #     print('{} epoch validation | test g_loss {:.4f} d_loss {:.4f}'.format(epoch + 1,
        #         test_g_loss / len(testloader.dataset), test_d_loss / len(testloader.dataset)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='mnist')
    parser.add_argument('--log_file', default='dagan_params')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--test_frequency', default=1, type=int)
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--z_dim', default=20, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    args = parser.parse_args()

    args.cuda = torch.cuda.device_count() != 0

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    enc = Encoder()
    dec = Decoder()
    generator = AEGenerator(enc, dec)
    discriminator = Discriminator()
    dagan = DAGAN(generator, discriminator)

    d_learning_rate = 2e-4
    g_learning_rate = 2e-4
    optim_betas = (0.9, 0.999)
    opt_g = torch.optim.Adam(generator.parameters(), lr=g_learning_rate, betas=optim_betas)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=d_learning_rate, betas=optim_betas)

    trainloader, testloader = utils.get_dataloaders(
        data=args.data, train_bs=args.batch_size, ohe_labels=True)

    train(trainloader, testloader, dagan, opt_g, opt_d, args)
