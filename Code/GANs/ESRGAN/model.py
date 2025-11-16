"""
The model is based on the paper "Enhanced Super-Resolution Generative Adversarial Networks"
by Xintao Wang, Ke Yu, Shixiang Wu, Jin-Yi Yang, Xiaoou Tang, and Leung-Shaumeng Lai

The model is a modified version of the SRGAN model
Basically, it improves on the residual block of the SRGAN model by using a more advanced
Residual-in-Residual Dense Blocks (RRDB) to better retain details. 
It also uses a Relativistic GAN (RaGAN) loss function for more realistic textures and 
improves the perceptual loss to reduce artifacts and improve sharpness. 

Additionally, ESRGAN removes the Batch Normalization layer to eliminate artifacts and 
uses residual scaling for training the very deep network. 
"""

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_act, **kwargs):
        super(ConvBlock, self).__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.cnn(x))
        

class UpSampleBlock(nn.Module):
    def __init__(self, in_c, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.conv = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, stride=1, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(self.upsample(x)))
        

class DenseResidualBlock(nn.Module):
    def __init__(self, in_channels, channels=32, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.blocks = nn.ModuleList() # ModuleList because we need to add multiple blocks

        for i in range(5):
            self.blocks.append(ConvBlock(
                in_channels + i * channels, 
                channels if i <= 3 else in_channels, 
                use_act=True if i <= 3 else False,
                kernel_size=3, padding=1, stride=1
                )   
            )
            # because we are concatenating the input channels with the output channels of the previous block
            # channels will be 32 unless the last part

    def forward(self, x):
        new_inputs = x
        for block in self.blocks: # because we are concatenating the input channels with the output channels of the previous block
            out = block(new_inputs)
            new_inputs = torch.cat((new_inputs, out), dim=1) # this is not needed but it makes things clear
        return self.residual_beta * out + x


# Residual-in-Residual Dense Block - RRDB
class RRDB(nn.Module):
    def __init__(self, in_channels, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.rrdb = nn.Sequential(
            DenseResidualBlock(in_channels, residual_beta=residual_beta),
            DenseResidualBlock(in_channels, residual_beta=residual_beta),
            DenseResidualBlock(in_channels, residual_beta=residual_beta),
        )

    def forward(self, x):
        return self.residual_beta * self.rrdb(x) + x
        

# Generator
class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_rrdb=23): # 10x the number of parameters of SRGAN
        super().__init__()
        self.initial = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, stride=1, bias=True) #SRGAN uses 9x9 kernel
        self.residual = nn.Sequential(*[RRDB(num_channels) for _ in range(num_rrdb)]) #23 times the RRDB block
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, stride=1)
        self.upsamples = nn.Sequential(
            UpSampleBlock(num_channels),
            UpSampleBlock(num_channels),
        )
        self.final = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, stride=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_channels, in_channels, kernel_size=3, padding=1, stride=1, bias=True),
        )

    def forward(self, x):
        initial = self.initial(x)
        x = self.conv(self.residual(initial)) + initial
        x = self.upsamples(x)
        x = self.final(x)
        return x


# same as SRGAN discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size=3, stride=1, padding=1, use_act=True),
            ConvBlock(64, 64, kernel_size=3, stride=2, padding=1, use_act=True),
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1, use_act=True),
            ConvBlock(128, 128, kernel_size=3, stride=2, padding=1, use_act=True),
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1, use_act=True),
            ConvBlock(256, 256, kernel_size=3, stride=2, padding=1, use_act=True),
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1, use_act=True),
            ConvBlock(512, 512, kernel_size=3, stride=2, padding=1, use_act=True),
        )
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.final(self.block(x))

        
def initialize_weights(model, scale=0.1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data)
            m.weight.data *= scale
        
    
def test():
    low_resolution = 24  # 96x96 -> 24x24
    with torch.cuda.amp.autocast():
        x = torch.randn((5, 3, low_resolution, low_resolution))
        gen = Generator()
        gen_out = gen(x)
        disc = Discriminator()
        disc_out = disc(gen_out)

        print(gen_out.shape)
        print(disc_out.shape)


if __name__ == "__main__":
    test()
        
        

        