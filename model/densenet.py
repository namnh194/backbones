import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    def forward(self, x):
        x = self.conv(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, ):
class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()

    def forward(self, x):
        return