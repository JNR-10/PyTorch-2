import torch
from torch import nn
from torchvision.models import vgg19
import config

# phi_5, 4 5th conv layer before maxpooling but after activation function

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features[:36].eval().to(config.DEVICE)
        self.loss = nn.MSELoss()
        
        for param in vgg.parameters():
            param.requires_grad = False
            
    def forward(self, gen_out, real_out):
        gen_features = self.vgg(gen_out)
        real_features = self.vgg(real_out)
        return self.loss(gen_features, real_features)
        