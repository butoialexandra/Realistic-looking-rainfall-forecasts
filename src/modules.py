"""
Code adapted from https://github.com/yuzhoucw/230pix2pix,
used in the paper https://cs230.stanford.edu/projects_spring_2018/reports/8289557.pdf.
The generator is the one giving the best performance in the paper (Encoder + 9 ResNet blocks
+ Decoder), including more deconv layers to increase resolution. The discriminator is the
ImageGAN discriminator, which uses a series on downsampling convolutional-BatchNorm-LeakyReLU
layers, modified to include the observations
"""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


# def norm_relu_layer(out_channel, do_norm, norm, relu):
#     if do_norm:
#         if norm == 'instance':
#             norm_layer = nn.InstanceNorm2d(out_channel)
#         elif norm == 'batch':
#             norm_layer = nn.BatchNorm2d(out_channel)
#         else:
#             raise NotImplementedError("norm error")
#     else:
#         norm_layer = nn.Dropout2d(0)  # Identity
#
#     if relu is None:
#         relu_layer = nn.ReLU()
#     else:
#         relu_layer = nn.LeakyReLU(relu, inplace=True)
#
#     return norm_layer, relu_layer


# def Conv_Norm_ReLU(in_channel, out_channel, kernel, padding=0, dilation=1, groups=1, stride=1, bias=True,
#                    do_norm=True, norm='batch', relu=None):
#     """
#     Convolutional -- Norm -- ReLU Unit
#     :param norm: 'batchnorm' --> use BatchNorm2D, 'instancenorm' --> use InstanceNorm2D, 'none' --> Identity()
#     :param relu: None -> Use vanilla ReLU; float --> Use LeakyReLU(relu)
#     :input (N x in_channel x H x W)
#     :return size same as nn.Conv2D
#     """
#     norm_layer, relu_layer = norm_relu_layer(out_channel, do_norm, norm, relu)
#
#     return nn.Sequential(
#         nn.Conv2d(in_channel, out_channel, kernel, padding=padding, stride=stride,
#                   dilation=dilation, groups=groups, bias=bias),
#         norm_layer,
#         relu_layer
#     )


class ResidualLayer(nn.Module):
    def __init__(self, channels, kernel_size=(3,3), stride=1, norm=None, activation="leakyrelu"):
        super().__init__()
        # compute padding
        self.padding = (kernel_size[0] - 1) // 2
        # first convolution
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                               stride=stride, padding=self.padding)
        # second convolution
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                               stride=1, padding=self.padding)
        if norm == "batch":
            self.batch_norm = nn.BatchNorm2d(channels)
        # ReLU or LeayReLU activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leakyrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        # list of layers
        self.model = []
        # first convolution -> batch norm/spectral norm -> activation
        if norm == "batch":
            self.model += [self.conv1, self.batch_norm, self.activation]
        elif norm == "spectral":
            self.model += [spectral_norm(self.conv1), self.activation]
        # second convolution -> batch norm/spectral norm -> activation
        if norm == "batch":
            self.model += [self.conv2, self.batch_norm, self.activation]
        elif norm == "spectral":
            self.model += [spectral_norm(self.conv2), self.activation]
        self.layers = nn.Sequential(*self.model)

    def forward(self, input):
        out = self.layers(input)
        return out + input


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False,
                 norm = None, activation = "leakyrelu"): # bias default is True in Conv2d
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leakyrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        if norm == "batch":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == "spectral":
            self.norm = spectral_norm(self.conv)
        else:
            self.norm = None

    def forward(self, x):
        if self.norm == "spectral":
            x = self.norm(x)
        elif self.norm == "batch":
            x = self.conv(x)
            x = self.norm(x)
        else:
            x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)

        return x


# def Deconv_Norm_ReLU(in_channel, out_channel, kernel, padding=0, output_padding=0, stride=1, groups=1,
#                      bias=True, dilation=1, do_norm=True, norm='batch'):
#     """
#     Deconvolutional -- Norm -- ReLU Unit
#     :param norm: 'batchnorm' --> use BatchNorm2D, 'instancenorm' --> use InstanceNorm2D, 'none' --> Identity()
#     :param relu: None -> Use vanilla ReLU; float --> Use LeakyReLU(relu)
#     :input (N x in_channel x H x W)
#     :return size same as nn.ConvTranspose2D
#     """
#     norm_layer, relu_layer = norm_relu_layer(out_channel, do_norm, norm, relu=None)
#     return nn.Sequential(
#         nn.ConvTranspose2d(in_channel, out_channel, kernel, padding=padding, output_padding=output_padding,
#                            stride=stride, groups=groups, bias=bias, dilation=dilation),
#         norm_layer,
#         relu_layer
#     )

class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=3, padding=1, output_padding=1, stride=1, groups=1,
                      bias=True, dilation=1, norm=None, activation='relu'):
        super(DecoderBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channel, out_channel, kernel, padding=padding, output_padding=output_padding,
                                       stride=stride, groups=groups, bias=bias, dilation=dilation)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leakyrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        if norm == "batch":
            self.norm = nn.BatchNorm2d(out_channel)
        elif norm == "spectral":
            self.norm = spectral_norm(self.conv)
        else:
            self.norm = None

    def forward(self, x):
        if self.norm == "spectral":
            x = self.norm(x)
        elif self.norm == "batch":
            x = self.conv(x)
            x = self.norm(x)
        else:
            x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, bias = False, norm = "spectral"):
        super(Discriminator, self).__init__()
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
        # final = nn.Sigmoid()(d13)
        return d13

class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, norm=None, bias=True):
        super(Generator, self).__init__()
        self.enc = nn.Sequential(*[
            EncoderBlock(in_channels, 62, stride=1, padding=1, bias=bias, norm=norm, activation="relu"),
            EncoderBlock(62, 124, stride=2, bias=bias, padding=1, norm=norm, activation="relu"),
            EncoderBlock(124, 248, stride=2, bias=bias, padding=1, norm=norm, activation="relu")
        ])
        modules = []
        for i in range(6):
            modules += [ResidualLayer(256, activation="relu")]
        modules += [DecoderBlock(256, 128, kernel=3, stride=2, bias=bias, norm=norm, activation='relu'),
                    DecoderBlock(128, 64, kernel=3, stride=2, bias=bias, norm=norm, activation='relu'),
                    DecoderBlock(64, 32, kernel=3, stride=2, bias=bias, padding=0, norm=norm, activation='relu'),
                    nn.Conv2d(32, out_channels, (3,3), stride=1, bias=bias),
                    nn.Sigmoid()]
        self.dec = nn.Sequential(*modules)

        # model = []
        # res = ResidualLayer(128, (3, 3), final_relu=False, bias=bias)
        # model += [Conv_Norm_ReLU(in_channels, 32, (7, 7), padding=3, stride=1, bias=bias, do_norm=do_norm, norm=norm),
        #           Conv_Norm_ReLU(32, 64, (3, 3), padding=1, stride=2, bias=bias, do_norm=do_norm, norm=norm),
        #           Conv_Norm_ReLU(64, 128, (3, 3), padding=1, stride=2, bias=bias, do_norm=do_norm, norm=norm)]
        # for i in range(3):
        #     model += [res]
        # model += [
        #     Deconv_Norm_ReLU(128, 64, (3, 3), padding=1, output_padding=1, stride=2, bias=bias, do_norm=do_norm, norm=norm),
        #     Deconv_Norm_ReLU(64, 32, (3, 3), padding=1, output_padding=1, stride=2, bias=bias, do_norm=do_norm, norm=norm),
        #     Deconv_Norm_ReLU(32, 16, (3, 3), padding=1, output_padding=1, stride=2, bias=bias, do_norm=do_norm,
        #                      norm=norm),
        #     nn.Conv2d(16, out_channels, (7, 7), padding=3, stride=1, bias=bias),
        #     nn.Tanh()]
        # self.model = nn.Sequential(*model)

    def forward(self, x, noise):
        e = self.enc(x)
        c = torch.cat([e, noise],1)
        d = self.dec(c)
        return d

class ResidualDenseBlock(nn.Module):
    def __init__(self, nf, gc=32, res_scale=0.2):
        super(ResidualDenseBlock, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(nf + 0 * gc, gc, 3, padding=1, bias=True), nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(nf + 1 * gc, gc, 3, padding=1, bias=True), nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(nf + 2 * gc, gc, 3, padding=1, bias=True), nn.LeakyReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(nf + 3 * gc, gc, 3, padding=1, bias=True), nn.LeakyReLU())
        self.layer5 = nn.Sequential(nn.Conv2d(nf + 4 * gc, nf, 3, padding=1, bias=True), nn.LeakyReLU())

        self.res_scale = res_scale

    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(torch.cat((x, layer1), 1))
        layer3 = self.layer3(torch.cat((x, layer1, layer2), 1))
        layer4 = self.layer4(torch.cat((x, layer1, layer2, layer3), 1))
        layer5 = self.layer5(torch.cat((x, layer1, layer2, layer3, layer4), 1))
        return layer5.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, nf, gc=32, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.layer1 = ResidualDenseBlock(nf, gc)
        self.layer2 = ResidualDenseBlock(nf, gc)
        self.layer3 = ResidualDenseBlock(nf, gc, )
        self.res_scale = res_scale

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out.mul(self.res_scale) + x


def upsample_block(nf, scale_factor=2):
    block = []
    for _ in range(scale_factor//2):
        block += [
            nn.Conv2d(nf, nf * (2 ** 2), 1),
            nn.PixelShuffle(2),
            nn.ReLU()
        ]

    return nn.Sequential(*block)


# class ESRGANDiscriminator(nn.Module):
#     def __init__(self, num_conv_block=4):
#         super(Discriminator, self).__init__()
#
#         block = []
#
#         in_channels = 3
#         out_channels = 64
#
#         for _ in range(num_conv_block):
#             block += [nn.ReflectionPad2d(1),
#                       nn.Conv2d(in_channels, out_channels, 3),
#                       nn.LeakyReLU(),
#                       nn.BatchNorm2d(out_channels)]
#             in_channels = out_channels
#
#             block += [nn.ReflectionPad2d(1),
#                       nn.Conv2d(in_channels, out_channels, 3, 2),
#                       nn.LeakyReLU()]
#             out_channels *= 2
#
#         out_channels //= 2
#         in_channels = out_channels
#
#         block += [nn.Conv2d(in_channels, out_channels, 3),
#                   nn.LeakyReLU(0.2),
#                   nn.Conv2d(out_channels, out_channels, 3)]
#
#         self.feature_extraction = nn.Sequential(*block)
#
#         self.avgpool = nn.AdaptiveAvgPool2d((512, 512))
#
#         self.classification = nn.Sequential(
#             nn.Linear(8192, 100),
#             nn.Linear(100, 1)
#         )
#
#     def forward(self, x):
#         x = self.feature_extraction(x)
#         x = x.view(x.size(0), -1)
#         x = self.classification(x)
#         return x

class ESRGAN(nn.Module):
    def __init__(self, in_channels, out_channels, nf=64, gc=32, scale_factor=2, n_basic_block=5):
        super(ESRGAN, self).__init__()

        self.conv1 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(in_channels, nf, 3), nn.ReLU())

        basic_block_layer = []

        for _ in range(n_basic_block):
            basic_block_layer += [ResidualInResidualDenseBlock(nf, gc)]

        self.basic_block = nn.Sequential(*basic_block_layer)

        self.conv2 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(nf, nf, 3), nn.ReLU())
        self.upsample = upsample_block(nf, scale_factor=scale_factor)
        self.conv3 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(nf, nf, 3), nn.ReLU())
        self.conv4 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(nf, out_channels, 3), nn.ReLU())
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.basic_block(x1)
        x = self.conv2(x)
        x = self.upsample(x + x1)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.activation(x)
        return x