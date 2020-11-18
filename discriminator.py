import torch
import torch.nn as nn
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        in_pixels = int(np.prod(input_shape))
        #out_pixels = int(np.prod(output_shape))

        self.model = nn.Sequential(
            # nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, stride=1),
            # nn.Linear(out_pixels + in_pixels, 512),
            nn.Linear(in_pixels, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, observation):
        # Concatenate label embedding and image to produce input
        # observation = observation.view(observation.size(0), -1)
        # print(observation.shape, img.shape)
        # d_in = torch.cat((img.view(img.size(0), -1), observation), -1)
        # d_in = torch.stack((observation, img), dim=1)
        d_in = img.view(img.size(0), -1)
        validity = self.model(d_in)
        validity = torch.sigmoid(validity)
        return validity
