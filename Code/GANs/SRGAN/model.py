import torch
from torch import nn


class ConvBlock(nn.Module):
    # Conv -> BatchNorm -> LeakyReLU

    def __init__(self, in_channels, out_channels, discriminator=False, use_act=True, use_bn=True, **kwargs):
        super().__init__()
        self.use_act = use_act
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True) if discriminator else nn.PReLU(num_parameters=out_channels)
        
    def forward(self, x):
        return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))
        


class UpSampleBlock(nn.Module):
    # Conv -> BatchNorm -> LeakyReLU -> Conv -> BatchNorm -> LeakyReLU
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * scale_factor ** 2, kernel_size=3, padding=1, stride=1)
        self.ps = nn.PixelShuffle(scale_factor) # in_channels, H, W --> in_channels, H*2, W*2
        self.act = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))
    


class ResidualBlock(nn.Module):
    # Conv -> BatchNorm -> LeakyReLU -> Conv -> BatchNorm
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = ConvBlock(in_channels, in_channels, kernel_size=3, padding=1, stride=1, use_act=False)
    
    def forward(self, x):
        return x + self.conv2(self.conv1(x)) # due to residual connection
    


class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=16):
        super().__init__()
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=9, padding=4, stride=1, use_bn = False)
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.convBlock = ConvBlock(num_channels, num_channels, kernel_size=3, padding=1, stride=1, use_act = False)
        self.upsamples = nn.Sequential(
            UpSampleBlock(num_channels, scale_factor=2),
            UpSampleBlock(num_channels, scale_factor=2)
        )
        self.final = ConvBlock(num_channels, in_channels, kernel_size=9, padding=4, stride=1)
        
    def forward(self, x):
        intial = self.initial(x)
        x = self.residuals(intial)
        x = self.convBlock(x) + intial
        x = self.upsamples(x)
        x = self.final(x)
        return torch.tanh(x)
        


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size=3, stride=1, padding=1, discriminator=True, use_bn=False),
            ConvBlock(64, 64, kernel_size=3, stride=2, padding=1, discriminator=True),
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1, discriminator=True),
            ConvBlock(128, 128, kernel_size=3, stride=2, padding=1, discriminator=True),
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1, discriminator=True),
            ConvBlock(256, 256, kernel_size=3, stride=2, padding=1, discriminator=True),
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1, discriminator=True),
            ConvBlock(512, 512, kernel_size=3, stride=2, padding=1, discriminator=True),
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
        