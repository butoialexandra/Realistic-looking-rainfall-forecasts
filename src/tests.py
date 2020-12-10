import torch


def test_generator():
    from generator import Generator

    t = torch.rand(4, 100)
    g = Generator(ngpu=1, noise_dim=100)
    t = g(t)
    assert t.shape[0] == 4

def test_discriminator():
    from discriminator import Discriminator

    t = torch.rand(4, 1, 128, 192)
    g = Discriminator(ngpu=1, n_channels=1)
    t = g(t)
    assert t.shape[0] == 4
