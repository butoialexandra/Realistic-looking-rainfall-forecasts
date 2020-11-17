import torch
import torch.nn as nn

import numpy as np

import util
from config import input_shape, output_shape


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            # if normalize:
                # layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.in_pixels = int(np.prod(input_shape))
        self.out_pixels = int(np.prod(output_shape))

        self.model = nn.Sequential(
            *block(1000 + 0
            # self.in_pixels
            , 1024, normalize=False),
            *block(1024, 1024),
            # *block(2560, 512),
            # *block(512, 1024),
            nn.Linear(1024, self.out_pixels),
            # nn.Tanh()
            nn.Sigmoid(),
        )

    def forward(self, noise, observation):
        # Concatenate label embedding and image to produce input
        # Keep only batch size and flatten everything else
        observation = observation.view(observation.size(0), self.in_pixels)#-1)
        # print("A",observation.size())
        # gen_input = torch.cat((observation, noise), -1)
        gen_input = noise
        # print("B",gen_input.size())
        img = self.model(gen_input)
        # print("C",img.size())
        img = img.view(img.size(0), *output_shape)
        # print("D",img.size())
        return img


class GeneratorA(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(*self._build_encoder())
        self.decoder = nn.Sequential(*self._build_decoder())
        self.alt_decoder = nn.Sequential(*self._build_alt_decoder())
        self.upsample = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=3, output_padding=(1,2))
        # self.input_dims = (127,188)

    def _build_encoder(self):
        layers = [
            # (batch_size, 1, 127, 188) -> 127x188
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1),
            # (batch_size, 1, h-2, w-2) -> 125x186
            nn.MaxPool2d(kernel_size=3, stride=3),
            # (batch_size, 1, floor((h-2)/3), floor((w-2)/3)) -> 41x62
            util.Flatten(),
            # (batch_size, 41*62) -> 2542
            nn.Linear(in_features=41*62, out_features=2604),
            nn.Sigmoid(),
        ]
        return layers

    def _build_alt_decoder(self):
        layers = [
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=3),
            # nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=1),
        ]
        return layers

    def _build_decoder(self):
        # Deconvolution:
        # output_size = stride * (input_size -1) + kernel_size - 2*pad
        layers = [
            # (batch_size, 1024)
            nn.Linear(in_features=1024, out_features=2623),
            nn.Sigmoid(),
            # util.View((1, 59, 85)), # and 5015?
            util.View((1, 43, 61)),  # for 2623
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=7, stride=7, padding=0)
            # 295, 427
        ]
        return layers

    def forward(self, noise, input):
        assert input.shape[1:] == input_shape, (input.shape, input_shape)
        embedding = self.encoder(input.unsqueeze(1))
        # assert embedding.shape[1] == 1024, embedding.shape
        # output = self.decoder(embedding).squeeze(1)
        # output = output[:,:295,:]  # we just have to get rid of 6 extra elements here
        embed_in = embedding.view(embedding.size(0),1,input.size(1)//3, input.size(2)//3)
        output = self.upsample(embed_in, output_size=input.size()[1:])

        # output = self.alt_decoder(input.unsqueeze(1)).squeeze(1)
        # output = output[:,:295, :427]
        # assert output.shape[1:] == output_shape, (output.shape, output_shape)
        # print(output.shape)
        return output.contiguous()
