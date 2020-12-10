import torch
import torch.nn as nn

from utils import Flatten


class Discriminator(nn.Module):

    """ This discriminator expects 128x192 input. Use with <forecast> data. """

    def __init__(self, ngpu, n_channels, n_filters=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.n_filters = n_filters
        self.n_channels = n_channels

        self.main = nn.Sequential(
            # input is (nc_disc) x 128 x 192
            nn.Conv2d(n_channels, n_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_filters) x 64 x 96
            nn.Conv2d(n_filters, n_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_filters) x 32 x 48
            nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_filters*2) x 16 x 24
            nn.Conv2d(n_filters * 2, n_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_filters*4) x 8 x 12
            nn.Conv2d(n_filters * 4, n_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_filters*8) x 4 x 6
            nn.Conv2d(n_filters * 8, 1, (4,6), 1, 0, bias=False),
            # Flatten(),
            # nn.Linear(35, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        # [64, 1, 256, 384] (-> [64, 1, 5, 7]) -> [64, 1]
        #print("Discrim in:", input.size())
        #print("Discrim out:", output.size())

        return output.view(-1, 1).squeeze(1)  # [batch_size]


class DiscriminatorHighres(nn.Module):

    """ This discriminator expects 256x384 input. Use with <observation> data. """

    def __init__(self, ngpu, n_channels, n_filters=64):
        super(DiscriminatorHighres, self).__init__()
        self.ngpu = ngpu
        self.n_filters = n_filters
        self.n_channels = n_channels

        self.main = nn.Sequential(
            # input is (nc_disc) x 128 x 192
            nn.Conv2d(n_channels, n_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_filters) x 64 x 96
            nn.Conv2d(n_filters, n_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_filters) x 32 x 48
            nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_filters*2) x 16 x 24
            nn.Conv2d(n_filters * 2, n_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_filters*4) x 8 x 12
            nn.Conv2d(n_filters * 4, n_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_filters*8) x 4 x 6
            nn.Conv2d(n_filters * 8, n_filters * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_filters*8) x 4 x 6
            nn.Conv2d(n_filters * 16, 1, (4,6), 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)
