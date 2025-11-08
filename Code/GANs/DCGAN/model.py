"""
From the paper Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
(https://arxiv.org/abs/1511.06434) by Alec Radford et al.

This used deep convolution networks instead of the fully connected networks used in the original GAN paper.

Generator:
takes 100-dim noise vector as input
and unsamples it to 4x4x1024 feature maps through a series of fractionally-strided convolutions (convtranspose2d)(also called deconvolutions)
until it reaches 64x64x3 image size.
BatchNorm is applied to all layers except the output layer, and ReLU is used as activation function except for the output layer which uses Tanh.

Discriminator: (exactly opposite of generator)
is a standard CNN that takes 64x64x3 image as input
and downsamples it to a single scalar output through a series of strided convolutions.
LeakyReLU is used as activation function and BatchNorm is applied to all layers except the input layer.

Look at the guidelines for stable DCGANS in the paper.
"""


import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, channels_img, feadures_d): # channels_img is 3 for RGB images, feadures_d is channels as we go through the layers of discriminator
        super (Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input: N x channels_img x 64 x 64
            nn.Conv2d(
                channels_img, feadures_d, kernel_size=4, stride=2, padding=1
            ),  # 32x32
            nn.LeakyReLU(0.2),
            self._block(feadures_d, feadures_d * 2, 4, 2, 1),  # 16x16
            self._block(feadures_d * 2, feadures_d * 4, 4, 2, 1),  # 8x8
            self._block(feadures_d * 4, feadures_d * 8, 4, 2, 1),  # 4x4
            nn.Conv2d(
                feadures_d * 8, 1, kernel_size=4, stride=1, padding=0
            ),  # Output: N x 1 x 1 x 1
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        return self.disc(x)
    

class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g): # z_dim is latent noise dim, channels_img is 3 for RGB images, features_g is channels as we go through the layers of generator
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            # Interpreted from the figure in the paper
            self._block(z_dim, features_g * 16, 4, 1, 0),  # 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 32x32
            nn.ConvTranspose2d(
                features_g * 2,
                channels_img,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.gen(x)
    

def initialize_weights(model):
    # Initialize model weights to follow normal distribution with mean=0 and std=0.02 as per DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1)
    gen = Generator(z_dim, in_channels, 8)
    initialize_weights(gen)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W)
    print("DCGAN tests passed")


if __name__ == "__main__":
    test()