"""

Below is a ResNet18 architecture:
Residual Network (ResNet) uses the concept of residual blocks that helps address the issue of vanishing
gradient in feedforward networks. The gradient in conventional deep learning networks is backpropagated to the previous
layers and the more deeper the network, the smaller the gradient. This can affect the performance of the networks.
So, one way of solving this problem was the introduction of shortcut connections to fully connected layers which simply
perform identity mapping (known as residual mapping). Some of the common variants of ResNets includes 18,34,50,101 and
152 which have stacks of residual or bottleneck blocks. The residual block comprises of two convolutional layers, and
it is used in ResNet18 and ResNet34, while the bottleneck has three convolutional layers for other variants. The bottleneck was introduced to allow fast training

"""
import torch.nn as nn
from torch.nn import functional as F

class Residual(nn.Module):
    def __init__(self, in_channel, out_channel, use_1x1Conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        if use_1x1Conv:
            self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, X):
        out = F.relu(self.bn1(self.conv1(X)))
        out = self.bn2(self.conv2(out))
        if self.conv3:
            X = self.conv3(X)
        out += X
        return F.relu(out)

def residualBlock(in_channel, out_channel, num_residuals, first_block=False):
    blks = []
    for i in range(num_residuals):
        if i==0 and not first_block:
            blks.append(Residual(in_channel, out_channel, use_1x1Conv=True,
            strides=2))
        else:
            blks.append(Residual(out_channel, out_channel))
    
    return blks



class ResNet(nn.Module):
    def __init__(self, input_channel, n_classes):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(*residualBlock(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*residualBlock(64, 128, 2))
        self.b4 = nn.Sequential(*residualBlock(128, 256, 2))
        self.b5 = nn.Sequential(*residualBlock(256, 512, 2))
        self.finalLayer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),nn.Linear(512, n_classes))

        self.b1.apply(self.init_weights)
        self.b2.apply(self.init_weights)
        self.b3.apply(self.init_weights)
        self.b4.apply(self.init_weights)
        self.b5.apply(self.init_weights)
        self.finalLayer.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Conv2d:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out')
        if type(layer) == nn.Linear:
            nn.init.normal_(layer.weight, std=1e-3)
        if type(layer) == nn.BatchNorm2d:
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)
        


    def forward(self, X):
        out = self.b1(X)
        out = self.b2(out)
        out = self.b3(out)
        out = self.b4(out)
        out = self.b5(out)
        out = self.finalLayer(out)

        return out