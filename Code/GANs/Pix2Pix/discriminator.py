import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        return self.conv(x)
        

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        # Initial layer is a simple conv layer without batch norm, as per the paper
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2, features[0], 4, 2, 1, padding_mode="reflect"), # in_channels*2 because we are concatenating the input and target
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]: # removes one redundant layer
            layers.append(CNNBlock(in_channels, feature, stride=1 if feature==features[-1] else 2))
            in_channels = feature

        layers.append(nn.Conv2d(in_channels, 1, 4, 1, 1, padding_mode="reflect"))
        
        self.model = nn.Sequential(*layers) # unpack the layers and put them in a sequential container

    
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x


def test():
    x = torch.randn(1, 3, 256, 256)
    y = torch.randn(1, 3, 256, 256)
    model = Discriminator()
    print(model(x, y).shape)

if __name__ == "__main__":
    test()
    
