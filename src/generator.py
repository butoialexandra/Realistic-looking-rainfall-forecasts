import torch
import torch.nn as nn

from utils import ListModule, Flatten


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

    def __init__(self, ngpu, input_dim, n_filters, skip_connections=False):
        super(CondGeneratorHighres, self).__init__()
        self.ngpu = ngpu
        self.input_dim = input_dim
        self.n_filters = n_filters
        self.use_skip_connections = skip_connections    

        self.encoder = ListModule(
            nn.Sequential(
                # input is (nc) x 128 x 192
                nn.Conv2d(1, n_filters, 4, 2, 1, bias=False),
                nn.BatchNorm2d(n_filters),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                # n_filters x 64 x 96
                nn.Conv2d(n_filters, n_filters, 4, 2, 1, bias=False),
                nn.BatchNorm2d(n_filters),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                # n_filters x 32 x 48
                nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(n_filters * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                # n_filters x 16 x 24
                nn.Conv2d(n_filters * 2, n_filters * 2, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # output: also 1 x 16 x 24 = 384
            ),
            nn.Sequential(
                # n_filters x 8 x 12
                nn.Conv2d(n_filters * 2, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # output: also 1 x 8 x 12 = 96
            )
        )

        decoder_filters = [
            n_filters * 32,
            n_filters * 16,
            n_filters * 8,
            n_filters * 4,
            n_filters * 2,
            n_filters,
            1
        ]
        
        decoder_input_channels = [
            96,
            n_filters * 32,
            n_filters * 16,
            n_filters * 8,
            n_filters * 4,
            n_filters * 2,
            n_filters
        ]

        if self.use_skip_connections:
            decoder_input_channels[2] += n_filters * 2
            decoder_input_channels[3] += n_filters * 2
            decoder_input_channels[4] += n_filters
            decoder_input_channels[5] += n_filters
            decoder_input_channels[6] += 1

        decoder_layers = []
        use_bias = True
        # Generate N blocks of deconvolution-batchnorm-nonlinearity
        for i, (in_channels, out_channels) in enumerate(zip(decoder_input_channels, decoder_filters)):
            block_layers = []
            if i == 0:
                # first layer has different parameters
                kernel_size = (4,6)
                stride, padding = 1, 0
            else:
                kernel_size, stride, padding = 4, 2, 1
            block_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=use_bias))
            if out_channels > 1:  # (i < len(decoder_filters) - 1):
                block_layers.append(nn.BatchNorm2d(out_channels))
                block_layers.append(nn.ReLU(inplace=True))
            else:
                # last layer can't have batchnorm, because it would output all zeros
                block_layers.append(nn.Tanh())
            decoder_layers.append(nn.Sequential(*block_layers))

        self.decoder = ListModule(*decoder_layers)

        #self.decoder_blocks = ListModule(
        #    nn.Sequential(
        #        # (384 = output of encoder) + (input_dim = 1,2)
        #        nn.ConvTranspose2d(96, n_filters * 32, (4,6), 1, 0, bias=False),
        #        nn.BatchNorm2d(n_filters * 32),
        #        nn.ReLU(True),
        #    ),
        #    nn.Sequential(
        #        # (n_filters*16) x 4 x 6
        #        nn.ConvTranspose2d(n_filters * 32, n_filters * 16, 4, 2, 1, bias=False),
        #        nn.BatchNorm2d(n_filters * 16),
        #        nn.ReLU(True),
        #    ),
        #    nn.Sequential(
        #        # (n_filters*8) x 8 x 12
        #        nn.ConvTranspose2d(n_filters * 16 + n_filters * 2, n_filters * 8, 4, 2, 1, bias=False),
        #        nn.BatchNorm2d(n_filters * 8),
        #        nn.ReLU(True),
        #    ),
        #    nn.Sequential(
        #        # (n_filters*4) x 16 x 24
        #        nn.ConvTranspose2d(n_filters * 8 + n_filters * 2, n_filters*4, 4, 2, 1, bias=False),
        #        nn.BatchNorm2d(n_filters * 4),
        #        nn.ReLU(True),
        #    ),
        #    nn.Sequential(
        #        # (n_filters*2) x 32 x 48
        #        nn.ConvTranspose2d(n_filters * 4 + n_filters,n_filters * 2, 4, 2, 1, bias=False),
        #        nn.BatchNorm2d(n_filters*2),
        #        nn.ReLU(True),
        #    ),
        #    nn.Sequential(
        #        # (n_filters) x 64 x 96
        #        nn.ConvTranspose2d(n_filters * 2 + n_filters, n_filters, 4, 2, 1, bias=False),
        #        nn.BatchNorm2d(n_filters),
        #        nn.ReLU(True),
        #    ),
        #    nn.Sequential(
        #        # (n_filters) x 128 x 192
        #        nn.ConvTranspose2d(n_filters + 1, 1, 4, 2, 1, bias=False),
        #        nn.Tanh()
        #        # output: 1 x 256 x 384
        #    )
        #)

    def forward(self, input, noise):
        noise = noise.unsqueeze(-1).unsqueeze(-1)

        # if input.is_cuda and self.ngpu > 1:
            # encoded = nn.parallel.data_parallel(self.encoder, input, range(self.ngpu))
            # encoded = encoded.view(encoded.size(0),encoded.size(1),1,1)
            # noisy_encoded = torch.cat((encoded,noise),1)
            # output = nn.parallel.data_parallel(self.decoded, noisy_encoded, range(self.ngpu))
        # else:
            # encoded = self.encoder(input)
            # encoded = encoded.view(encoded.size(0), encoded.size(1),1,1)
            # noisy_encoded = torch.cat((encoded, noise),1)
            # output = self.decoder(noisy_encoded)

        # TODO(mzilinec): fix for multiple GPUs
        
        output = self._forward(input, noise)
        return output

    def _forward(self, cond_input, noise):
        skip_conn_tgts = [2, 3, 4, 5, 6]
        skip_conns = []
        x = cond_input

        for block in self.encoder:
            # create skip connection to decoder
            if self.use_skip_connections:
                skip_conns.append(x)
            # apply encoder block
            x = block(x)

        x = torch.flatten(x, start_dim=1)
        x = x.view(x.size(0), x.size(1), 1, 1)
        # x = torch.cat((x, noise), dim=1)  # TODO(mzilinec): i DISABLED noise (see pix2pix)
        # TODO(mzilinec): the 2nd /noise/ dim likely gets ignored, we should *add* them instead

        for i, block in enumerate(self.decoder):
            # apply skip connection from encoder
            if self.use_skip_connections and i == skip_conn_tgts[0]:
                skip_conn_tgts = skip_conn_tgts[1:]
                skip_conn = skip_conns.pop()
                x = torch.cat((x, skip_conn), dim=1)
                # print(x.size())
            # apply decoder block
            x = block(x)
        
        return x
