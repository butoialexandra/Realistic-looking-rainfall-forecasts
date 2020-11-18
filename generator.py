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
