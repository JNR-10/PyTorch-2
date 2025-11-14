"""
Pros:
Better Stability
Loss means something: provides a termination criterion, previously loss told us nothing

Cons:
Longer to train (but stable and will let to less tweaking of hyperparameters)

Problem this solves is mode collapse where generator produces same output all the time

The idea in GAN is we have two probability distributions:
Pg and Pr
Pg is the distribution of generated data (implicit distribution)
Pr is the distribution of real data (explicit distribution)
We want these to be very similar and when we succed in that GAN converged and we generrat more realistic images

How do we define a distance between two probability distributions?
There are many ways to do this:
1. KL Divergence (Kullback-Leibler Divergence)
2. JS Divergence (used in original GAN paper)
3. Wasserstein Distance (Earth Mover Distance) - used in WGAN

JS Divergence has gradient issues leading to unstable training, and WGAN instead bases its loss from
Wasserstein Distance which provides smooth gradients everywhere

After a lot of math, we arrive at the WGAN loss functions:
Discriminator Loss:
    max E[ D(x) ] - E[ D(G(z)) ], when trained this will go to 0
Generator Loss:
    min - E[ D(G(z)) ]

Discriminator wants to maximize the difference between its output for real images and fake images
Generator wants to minimize the output of discriminator for fake images

Discriminator and Generator implementation from DCGAN paper,
with removed Sigmoid() as output from Discriminator (and therefor
it should be called critic)
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
            nn.Tanh(), # to get output between -1 and 1
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
    print("Success!")


if __name__ == "__main__":
    test() 