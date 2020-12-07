"""
Importing header files
"""
import torch
import torch.nn as nn
import train
from modules import EncoderBlock

ngpu = train.ngpu
nz = train.nz
ngf = train.ngf
ndf = train.ndf


class Discriminator(nn.Module):
    """
    Abstract class for Discriminator
    for low resolution images.
    """
    def __init__(self, ngpu):
        """
        Parameters
        ----------
        ngpu: int
            Number of GPUs
        """
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 128 x 192
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 96
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 48
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 24
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 12
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 6
            nn.Conv2d(ndf * 8, 1, (4,6), 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        Runs the model

        Parameters
        ----------
        input: ndarray of size (128, 192)
        """
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


class DiscriminatorHighres(nn.Module):
    """
    Abstract class for Discriminator
    for high resolution images.
    """
    def __init__(self, ngpu):
        super(DiscriminatorHighres, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 256 x 384
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 192
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 96
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 48
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 24
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 12
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 6
            nn.Conv2d(ndf * 16, 1, (4,6), 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        Runs the model

        Parameters
        ----------
        input: ndarray of size (256, 384)
        """
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


class DiscriminatorRes(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, bias = False, norm = "spectral"):
        super(DiscriminatorRes, self).__init__()
        self.disc1 = EncoderBlock(in_channels, 32, stride=2, bias=bias, norm=norm)
        self.disc2 = EncoderBlock(in_channels, 32, stride=2, bias=bias, norm=norm)
        self.disc3 = EncoderBlock(32, 64, stride=2, bias=bias, norm=norm)
        self.disc4 = EncoderBlock(32, 64, stride=2, bias=bias, norm=norm)
        self.disc5 = EncoderBlock(64, 128, stride=2, bias=bias, norm=norm)
        self.disc6 = EncoderBlock(64, 128, stride=2, bias=bias, norm=norm)
        self.disc7 = EncoderBlock(128, 256, stride=2, bias=bias, norm=norm)
        self.disc8 = EncoderBlock(128, 256, stride=1, bias=bias, norm=norm)
        self.disc9 = EncoderBlock(512, 512, stride=1, bias=bias, norm=norm)
        self.disc10 = EncoderBlock(512, out_channels, stride=1, bias=bias, norm=norm)
        self.pool = nn.AvgPool2d(14)

    def forward(self, x, ref):
        d1 = self.disc1(x)
        d2 = self.disc2(ref)
        d3 = self.disc3(d1)
        d4 = self.disc4(d2)
        d5 = self.disc5(d3)
        d6 = self.disc6(d4)
        d7 = self.disc7(d5)
        d8 = self.disc8(d6)
        d9 = torch.cat([d7, d8],1)
        d10 = self.disc9(d9)
        d11 = self.disc10(d10)
        d12 = self.pool(d11)
        d13 = d12.squeeze(3).squeeze(2)
        final = nn.Sigmoid()(d13)
        return final