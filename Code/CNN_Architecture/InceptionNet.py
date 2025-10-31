"""
InceptionNet Architecture Implementation
The main idea of InceptionNet was to select which kernel size to use at each layer
based on the input data. This is achieved by using multiple convolutional filters of
different sizes in parallel and concatenating their outputs. This allows the network
to learn which filter size is most appropriate for the given input, improving performance
and efficiency.

The idea was to have a lower dimentiona convolution (1x1) before the expensive convolutions (3x3 and 5x5)
to reduce the depth of the input volume (number of filters), thus reducing the computational cost. 

They also used auxiliary classifiers (avgpool->conv->fc->fc->softmax) connected to intermediate layers 
to help with gradient flow during training (avoid overfitting). 
This acts as a form of regularization and helps to improve convergence.
"""


import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs): #keyword arguments: kernel_size, stride, padding
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels) #not in paper cause it was 2014, but helps with training performance
        

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x))) #one -> after -> other


class Inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool): #from the table in the paper
        super(Inception_block, self).__init__()

        # from the diagram (flow) in the paper
        # 1x1 conv branch
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1) #stride and padding default to 1 and 0 respectively

        # 1x1 conv -> 3x3 conv branch
        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=1),
            conv_block(red_3x3, out_3x3, kernel_size=3, padding=1) #stride defaults to 1
        )

        # 1x1 conv -> 5x5 conv branch
        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=1),
            conv_block(red_5x5, out_5x5, kernel_size=5, padding=2)
        )

        # 3x3 maxpool -> 1x1 conv branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out_1x1pool, kernel_size=1)
        )

    def forward(self, x):
        # N x filters x 28 x 28 (example)
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)
# concatenate along the channel dimension in an array of outputs from each branch


class GoogleNet(nn.Module): #from the table (moving vertically downwards) in the paper
    def __init__(self, in_channels=3, num_classes=1000):
        super(GoogleNet, self).__init__()

        self.conv1 = conv_block(in_channels, 64, kernel_size=7, stride=2, padding=3) #N x 64 x 112 x 112
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #N x 64 x 56 x 56
        self.conv2 = conv_block(64, 192, kernel_size=3, stride=1, padding=1) #`N x 192 x 56 x 56
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #N x 192 x 28 x 28

        # In this order: `in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
        self.inception3a = Inception_block(192, 64, 96, 128, 16, 32, 32) #N x 256 x 28 x 28
        self.inception3b = Inception_block(256, 128, 128, 192, 32, 96, 64) #N x 480 x 28 x 28
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #N x 480 x 14 x 14

        self.inception4a = Inception_block(480, 192, 96, 208, 16, 48, 64) #N x 512 x 14 x 14
        self.inception4b = Inception_block(512, 160, 112, 224, 24, 64, 64) #N x 512 x 14 x 14
        self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64) #N x 512 x 14 x 14
        self.inception4d = Inception_block(512, 112, 144, 288, 32, 64, 64) #N x 528 x 14 x 14
        self.inception4e = Inception_block(528, 256, 160, 320, 32, 128, 128) #N x 832 x 14 x 14
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #N x 832 x 7 x 7

        self.inception5a = Inception_block(832, 256, 160, 320, 32, 128, 128) #N x 832 x 7 x 7
        self.inception5b = Inception_block(832, 384, 192, 384, 48, 128, 128) #N x 1024 x 7 x 7
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1) #N x 1024 x 1 x 1

        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)

        x = x.reshape(x.shape[0], -1)  # Flatten
        x = self.dropout(x)
        x = self.fc(x)
        return x
    

if __name__ == "__main__":
    model = GoogleNet(in_channels=3, num_classes=1000)
    x = torch.randn(3, 3, 224, 224) # 3 sample images with 3 channels of 224x224 dimensions each
    print(model(x).shape)  # Should output torch.Size([3, 1000])


