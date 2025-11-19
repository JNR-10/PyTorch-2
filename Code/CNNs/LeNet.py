import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions

"""
# Summary of the LeNet architecture:
# Input: 32x32x1 grayscale image, 
# C1: 5x5 kernal, stride 1, padding 0 (Output: 28x28x6)
# Average pool with kernel size 2 and stride 2 (Output: 14x14x6)
# 5x5 convolution layer stride 1, padding 0 (Output: 10x10x16)
# Average pooling layers with kernel size 2 and stride 2 (Output: 5x5x16)
# 5x5 conv layer stride 1, padding 0 (Output: 1x1x120) 
# Fully connected layer with 84 units
# Output layer with 10 units (for 10 classes)
"""


class LeNet(nn.Module):
    def __init__(self):
        # first call the constructor of the parent class nn.Module
        super(LeNet, self).__init__()

        # defining the layers
        self.relu = nn.ReLU() # Activation function
        self.pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2)) # Average Pooling layer
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0,
        )
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=5,
            stride=1,
            padding=0,
        )
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)

    # defining the forward pass
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(
            self.conv3(x)
        )  # num_examples x 120 x 1 x 1 --> num_examples x 120
        x = x.reshape(x.shape[0], -1) # flatten the tensor
        x = self.relu(self.linear1(x))
        x = self.linear2(x) # no activation function in output layer
        return x


def test_lenet():
    x = torch.randn(64, 1, 32, 32)
    model = LeNet()
    return model(x) # Output shape should be (64, 10)
    # 64 images with 10 probabilities each


if __name__ == "__main__":
    out = test_lenet()
    print(out.shape)


