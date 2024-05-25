import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, LeakyReLU, BatchNorm1d, Tanh, Sigmoid, \
    BCELoss
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

device = torch.device('cpu')


class Generator(nn.Module):
    """
    Linear args.latent_dim -> 128; LeakyReLU(0.2) \n
    Linear 128 -> 256; BatchNorm; LeakyReLU(0.2) \n
    Linear 256 -> 512; BatchNorm; LeakyReLU(0.2) \n
    Linear 512 -> 1024; BatchNorm; LeakyReLU(0.2) \n
    Linear 1024 -> 768 \n
    Output non-linearity \n
    """

    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.layers = Sequential(
            *[Linear(latent_dim, 128), LeakyReLU(0.2)],
            *[Linear(128, 256), BatchNorm1d(256), LeakyReLU(0.2)],
            *[Linear(256, 512), BatchNorm1d(512), LeakyReLU(0.2)],
            *[Linear(512, 1024), BatchNorm1d(1024), LeakyReLU(0.2)],
            Linear(1024, 784), Tanh()
        )

    def forward(self, z):
        x = self.layers(z)
        x = x.view(x.shape[0], 28, 28)
        return x


class Discriminator(nn.Module):
    """
    Linear 784 -> 512; LeakyReLU(0.2) \n
    Linear 512 -> 256; LeakyReLU(0.2) \n
    Linear 256 -> 1 \n
    Output non-linearity \n
    """

    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()
        self.layers = Sequential(
            *[Linear(784, 512), LeakyReLU(0.2)],
            *[Linear(512, 256), LeakyReLU(0.2)],
            Linear(256, 1), Sigmoid()
        )

    def forward(self, img):
        x = img.view(img.shape[0], 784)
        x = self.layers(x)
        x = x.squeeze()
        return x


def train(dataloader, discriminator, generator, optimizer_gen, optimizer_dis):
    log_likelihood = BCELoss()  # Binary Cross Entropy Loss

    for epoch in range(args.n_epochs):
        g_ll, d_ll = None, None
        for i, (imgs_real, _) in enumerate(dataloader):
            imgs_real = imgs_real.to(device)
            size = imgs_real.shape[0]
            optimizer_gen.zero_grad()
            optimizer_dis.zero_grad()
            label_real = torch.ones(size, dtype=torch.float, device=device)
            label_fake = torch.zeros(size, dtype=torch.float, device=device)

            # --- Train Generator ---
            z = torch.randn(size=(size, args.latent_dim), device=device)  # noise
            imgs_fake = generator(z)
            score_imgs_fake = discriminator(imgs_fake)
            g_ll = log_likelihood(score_imgs_fake, label_real)  # attention here
            g_ll.backward()
            optimizer_gen.step()

            # --- Train Discriminator ---
            score_imgs_fake = discriminator(imgs_fake.detach())
            score_imgs_real = discriminator(imgs_real.detach())
            d_ll_fake = log_likelihood(score_imgs_fake, label_fake)
            d_ll_real = log_likelihood(score_imgs_real, label_real)
            d_ll = (d_ll_fake + d_ll_real) / 2.
            d_ll.backward()
            optimizer_dis.step()

            # --- Save Images ---
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                gen_imgs = imgs_fake.unsqueeze(1)[:25]
                path = f'images/{batches_done}.png'
                save_image(gen_imgs, path,
                           nrow=5, normalize=True, value_range=(-1, 1))
                print(f'./{path}', end=' ')

        # --- Print Result ---
        print('\nEpoch: {}/{}, Generator Log Likelihood: {:.4f}, Discriminator Log Likelihood: {:.4f}, '
              .format(epoch + 1, args.n_epochs, g_ll.item(), d_ll.item()), end='')


def main():
    os.makedirs('./images', exist_ok=True)

    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,),
                                                (0.5,))])),
        batch_size=args.batch_size, shuffle=True)
    generator = Generator(latent_dim=args.latent_dim)
    discriminator = Discriminator(latent_dim=args.latent_dim)
    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    train(dataloader, discriminator, generator, optimizer_gen, optimizer_dis)

    torch.save(generator.state_dict(), 'mnist_generator.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500, help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
