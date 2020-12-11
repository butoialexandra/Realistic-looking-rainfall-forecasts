import torch


def test_generator():
    from generator import Generator, CondGeneratorHighres

    t = torch.rand(4, 100)
    g = Generator(ngpu=1, noise_dim=100)
    out = g(t)
    assert out.shape[0] == 4

    cg = CondGeneratorHighres(ngpu=1, input_dim=100, n_filters=64)

    i = torch.rand(4, 1, 128, 192)
    out = cg(input=i, noise=t)
    assert out.shape[0] == 4


def test_discriminator():
    from discriminator import Discriminator, DiscriminatorHighres
    from utils import upscale_input

    t = torch.rand(4, 1, 128, 192)
    g = Discriminator(ngpu=1, n_channels=1)
    out = g(t)
    assert out.shape[0] == 4

    i = torch.rand(4, 1, 128, 192)  # input we're conditioning on
    j = torch.rand(4, 1, 256, 384)  # output from generator or ground truth

    i = upscale_input(i)
    k = torch.cat((i, j), dim=1)
    
    dh = DiscriminatorHighres(ngpu=1, n_channels=2)
    out = dh(k)
    assert out.shape[0] == 4
