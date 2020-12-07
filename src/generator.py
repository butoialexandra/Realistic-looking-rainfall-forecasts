"""
Importing header files
"""
import torch.nn as nn

import train

ngpu = train.ngpu
nz = train.nz
ngf = train.ngf
ndf = train.ndf


class Generator(nn.Module):
    """
    Abstract class for generator of low resolution observations
    """
    def __init__(self, ngpu):
        """
        Parameters
        ----------
        ngpu: int
            Number of GPUs
        """

        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 16, (4,6), 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 6
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 12
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 24
            nn.ConvTranspose2d(ngf * 4,     ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 48
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 96
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 192
        )

    def forward(self, input):
        """
        Runs the model

        Parameters
        ----------
        input: ndarray of size (nz,)
        """

        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class GeneratorHighres(nn.Module):
    """
    Abstract class for the generator model
    to generate high resolution observations
    """
    def __init__(self, ngpu):
        """
        Parameters
        ----------
        ngpu: int
            Number of GPUs
        """

        super(GeneratorHighres, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution                                                                                         [381/652]
            nn.ConvTranspose2d(     nz, ngf * 32, (4,6), 1, 0, bias=False),
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
            # state size. (ngf*2) x 64 x 96
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 128 x 192
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 256 x 384
        )

    def forward(self, input):
        """
        Runs the model

        Parameters
        ----------
        input: ndarray of size (nz,)
        """
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output



class CondGenerator(nn.Module):
    """
    Abstract class for the generator model
    to generate low resolution observations
    conditioned on predictions
    """
    def __init__(self, ngpu):
        """
        Parameters
        ----------
        ngpu: int
            Number of GPUs
        """
        super(CondGenerator, self).__init__()
        self.ngpu = ngpu
        #Encoder
        self.encoder = nn.Sequential(
            # input is (nc) x 128 x 192
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 96
            nn.Conv2d(ndf, ndf, 4, 2, 1, biasFalse),                                                                                                                                              [336/652]
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 48
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            # state size. (ndf*2) x 16 x 24
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, 1, 1, bias=False),
            nn.Flatten()

        )
        #Decoder
        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(384+nz, ngf * 16, (4,6), 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 6
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 12
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 24
            nn.ConvTranspose2d(ngf * 4,     ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 48
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 96
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 192
)                                                                                                                                                                                          [289/652]

    def forward(self, input, noise):
        """
        Runs the model

        Parameters
        ----------
        input: ndarray of size (128,192)
        noise: ndarray of size (nz,)
        """
        if input.is_cuda and self.ngpu > 1:
            #Encoder
            encoded = nn.parallel.data_parallel(self.encoder, input, range(self.ngpu))
            encoded = encoded.view(encoded.size(0), encoded.size(1), 1, 1)
            #Noise
            noisy_encoded = torch.cat((encoded,noise),1)
            #Decoder
            output = nn.parallel.data_parallel(self.decoded, noisy_encoded, range(self.ngpu))
        else:
            #Encoder
            encoded = self.encoder(input)
            encoded = encoded.view(encoded.size(0), encoded.size(1), 1, 1)
            #Noise
            noisy_encoded = torch.cat((encoded, noise),1)
            #Decoder
            output = self.decoder(noisy_encoded)
        return output


class CondGeneratorHighres(nn.Module):
    """
    Abstract class for the generator model
    to generate low resolution observations
    conditioned on predictions
    """
    def __init__(self, ngpu):
        """
        Parameters
        ----------
        ngpu: int
            Number of GPUs
        """
        super(CondGeneratorHighres, self).__init__()
        self.ngpu = ngpu
        #Encoder
        self.encoder = nn.Sequential(
            # input is (nc) x 128 x 192
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 96
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 48
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            # state size. (ndf) x 16 x 24
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, 1, 1, bias=False),
            #add flatten layer
            nn.Flatten()

        )
        #Decoder
        self.decoder = nn.Sequential(                                                                                                                                                              [241/652]
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
            # state size. (ngf) x 128 x 192
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 256 x 384
        )

    def forward(self, input, noise):
        if input.is_cuda and self.ngpu > 1:
            #Encoder
            encoded = nn.parallel.data_parallel(self.encoder, input, range(self.ngpu))
            encoded = encoded.view(encoded.size(0),encoded.size(1),1,1)
            #Noise
            noisy_encoded = torch.cat((encoded,noise),1)
            #Decoder
            output = nn.parallel.data_parallel(self.decoded, noisy_encoded, range(self.ngpu))
        else:
            #Encoder
            encoded = self.encoder(input)
            encoded = encoded.view(encoded.size(0), encoded.size(1),1,1)
            #Noise
            noisy_encoded = torch.cat((encoded, noise),1)
            #Decoder
            output = self.decoder(noisy_encoded)
        return output






# Add Wasserstein generator
# Add Resnet generator
# Yinghao model
