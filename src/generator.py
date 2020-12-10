import torch
import torch.nn as nn


class Generator(nn.Module):

    """ This generator provides 128x192 output. Use with <forecast> data. """

    def __init__(self, ngpu, noise_dim, n_filters=64):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.noise_dim = noise_dim
        self.n_filters = n_filters

        self.main = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, n_filters * 16, (4,6), 1, 0, bias=False),
            nn.BatchNorm2d(n_filters * 16),
            nn.ReLU(True),
            # state size. (n_filters*16) x 4 x 4
            nn.ConvTranspose2d(n_filters * 16, n_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 8),
            nn.ReLU(True),
            # state size. (n_filters*8) x 8 x 8
            nn.ConvTranspose2d(n_filters * 8, n_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 4),
            nn.ReLU(True),
            # state size. (n_filters*4) x 16 x 16
            nn.ConvTranspose2d(n_filters * 4,     n_filters*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters*2),
            nn.ReLU(True),
            # state size. (n_filters*2) x 32 x 32
            nn.ConvTranspose2d(n_filters * 2,     n_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(True),
            # state size. (n_filters) x 64 x 64
            nn.ConvTranspose2d(n_filters, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        input = input.unsqueeze(-1).unsqueeze(-1)  # Convert 1D noise to 1x1 image
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        print("Gener in:", input.size())
        print("Gener out:", output.size())
        return output


class GeneratorHighres(nn.Module):

    """ This generator provides 256x384 output. Use with <observation> data. """

    def __init__(self, ngpu, noise_dim, n_filters=64):
        super(GeneratorHighres, self).__init__()
        self.ngpu = ngpu
        self.noise_dim = noise_dim
        self.n_filters = n_filters

        self.main = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, n_filters * 32, (4,6), 1, 0, bias=False),
            nn.BatchNorm2d(n_filters * 32),
            nn.ReLU(True),
            # state size. (n_filters*16) x 4 x 4
            nn.ConvTranspose2d(n_filters * 32, n_filters * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 16),
            nn.ReLU(True),
            # state size. (n_filters*8) x 8 x 8
            nn.ConvTranspose2d(n_filters * 16, n_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 8),
            nn.ReLU(True),
            # state size. (n_filters*4) x 16 x 16
            nn.ConvTranspose2d(n_filters * 8,     n_filters*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters*4),
            nn.ReLU(True),
            # state size. (n_filters*2) x 32 x 32
            nn.ConvTranspose2d(n_filters * 4,     n_filters*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters*2),
            nn.ReLU(True),
            # state size. (n_filters*2) x 32 x 32
            nn.ConvTranspose2d(n_filters * 2,     n_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(True),
            # state size. (n_filters) x 64 x 64
            nn.ConvTranspose2d(n_filters, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        input = input.unsqueeze(-1).unsqueeze(-1)  # Convert 1D noise to 1x1 image
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        print("Gener in:", input.size())
        print("Gener out:", output.size())
        return output


class CondGenerator(nn.Module):

    # TODO

    def __init__(self, ngpu, input_dim, n_filters):
        super(CondGenerator, self).__init__()
        self.ngpu = ngpu
        self.input_dim = input_dim
        self.n_filters = n_filters
        ndf = n_filters

        self.encoder = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(input_dim, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, 1, 1, bias=False),
            nn.Flatten()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(384+nz, n_filters * 16, (4,6), 1, 0, bias=False),
            nn.BatchNorm2d(n_filters * 16),
            nn.ReLU(True),
            # state size. (n_filters*16) x 4 x 6
            nn.ConvTranspose2d(n_filters * 16, n_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 8),
            nn.ReLU(True),
            # state size. (n_filters*8) x 8 x 12
            nn.ConvTranspose2d(n_filters * 8, n_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 4),
            nn.ReLU(True),
            # state size. (n_filters*4) x 16 x 24
            nn.ConvTranspose2d(n_filters * 4,     n_filters*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters*2),
            nn.ReLU(True),
            # state size. (n_filters*2) x 32 x 48
            nn.ConvTranspose2d(n_filters * 2,     n_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(True),
            # state size. (n_filters) x 64 x 96
            nn.ConvTranspose2d(    n_filters,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 192
        )

    def forward(self, input, noise):
        if input.is_cuda and self.ngpu > 1:
            encoded = nn.parallel.data_parallel(self.encoder, input, range(self.ngpu))
            encoded = encoded.view(encoded.size(0), encoded.size(1), 1, 1)
            noisy_encoded = torch.cat((encoded,noise),1)
            output = nn.parallel.data_parallel(self.decoded, noisy_encoded, range(self.ngpu))
        else:
            encoded = self.encoder(input)
            encoded = encoded.view(encoded.size(0), encoded.size(1), 1, 1)
            noisy_encoded = torch.cat((encoded, noise),1)
            output = self.decoder(noisy_encoded)
        return output


class CondGeneratorHighres(nn.Module):

    # TODO

    def __init__(self, ngpu, input_dim, n_filters):
        super(CondGeneratorHighres, self).__init__()
        self.ngpu = ngpu
        self.input_dim = input_dim
        self.n_filters = n_filters

        self.encoder = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(input_dim, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, 1, 1, bias=False),
            #add flatten layer
            nn.Flatten()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(384+nz, n_filters * 32, (4,6), 1, 0, bias=False),
            nn.BatchNorm2d(n_filters * 32),
            nn.ReLU(True),
            # state size. (n_filters*16) x 4 x 6
            nn.ConvTranspose2d(n_filters * 32, n_filters * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 16),
            nn.ReLU(True),
            # state size. (n_filters*8) x 8 x 12
            nn.ConvTranspose2d(n_filters * 16, n_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 8),
            nn.ReLU(True),
            # state size. (n_filters*4) x 16 x 24
            nn.ConvTranspose2d(n_filters * 8,     n_filters*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters*4),
            nn.ReLU(True),
            # state size. (n_filters*2) x 32 x 48
            nn.ConvTranspose2d(n_filters * 4,     n_filters*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters*2),
            nn.ReLU(True),
            # state size. (n_filters) x 64 x 96
            nn.ConvTranspose2d(n_filters * 2,     n_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(True),
            # state size. (n_filters) x 64 x 96
            nn.ConvTranspose2d(    n_filters,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 192
        )

    def forward(self, input, noise):
        if input.is_cuda and self.ngpu > 1:
            encoded = nn.parallel.data_parallel(self.encoder, input, range(self.ngpu))
            encoded = encoded.view(encoded.size(0),encoded.size(1),1,1)
            noisy_encoded = torch.cat((encoded,noise),1)
            output = nn.parallel.data_parallel(self.decoded, noisy_encoded, range(self.ngpu))
        else:
            encoded = self.encoder(input)
            encoded = encoded.view(encoded.size(0), encoded.size(1),1,1)
            noisy_encoded = torch.cat((encoded, noise),1)
            output = self.decoder(noisy_encoded)
        return output
