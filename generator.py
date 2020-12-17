import torch
import torch.nn as nn

import numpy as np

import util



class FeedforwardGenerator(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.input_shape = input_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.in_pixels = int(np.prod(input_shape))

        self.model = nn.Sequential(
            *block(self.in_pixels + latent_dim, 1024, normalize=False),
            *block(1024, self.in_pixels),
            nn.Sigmoid()
        )

    def forward(self, noise, observation):
        # Concatenate label embedding and image to produce input
        # Keep only batch size and flatten everything else
        observation = observation.view(observation.size(0), -1)
        gen_input = torch.cat((observation, noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), 1, *self.input_shape)
        return img


class ConvGenerator(nn.Module):

    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.input_shape = input_shape
        self.encoder = nn.Sequential(*self._build_encoder())
        self.upsample = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=3, output_padding=(1,2))

    def _build_encoder(self):
        layers = [
            # (batch_size, 1, 127, 188) -> 127x188
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1),
            # (batch_size, 1, h-2, w-2) -> 125x186
            nn.MaxPool2d(kernel_size=3, stride=3),
            # (batch_size, 1, floor((h-2)/3), floor((w-2)/3)) -> 41x62
            util.Flatten(),
            # (batch_size, 41*62) -> 2542
            nn.Linear(in_features=41*62, out_features=np.prod([self.input_shape[0]//3, self.input_shape[1]//3])),
            nn.Sigmoid(),
        ]
        return layers



    def forward(self, noise, input):
        assert input.shape[1:] == self.input_shape, (input.shape, self.input_shape)

        embedding = self.encoder(input.unsqueeze(1))
        embed_in = embedding.view(embedding.size(0),1,input.size(1)//3, input.size(2)//3)
        output = self.upsample(embed_in, output_size=input.size()[1:])
        return output.contiguous()


class CondGeneratorHighres(nn.Module):
    def __init__(self, ngpu, nz, ngf, ndf):
        super(CondGeneratorHighres, self).__init__()
        self.ngpu = ngpu
        self.encoder = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
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
            #nn.BatchNorm2d(ndf * 2),
            #nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            #nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 4),
            #nn.LeakyReLU(0.2, inplace=True),
            ## state size. (ndf*4) x 8 x 8
            #nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 8),
            #nn.LeakyReLU(0.2, inplace=True),
            ## state size. (ndf*8) x 4 x 4
            #nn.Conv2d(ndf * 8, 1, (4,6), 1, 0, bias=False),
            #nn.LeakyReLU(0.2, inplace=True)
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(384+nz, ngf * 32, (4,6), 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 32),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 6
            nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 12
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 24
            nn.ConvTranspose2d(ngf * 8,     ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 48
            nn.ConvTranspose2d(ngf * 4,     ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 96
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 96
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
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

