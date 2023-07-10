import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.downsample = downsample
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet34(nn.Module):
    def __init__(self, block, num_classes=10):
        super(ResNet34, self).__init__()
        self.num_classes = num_classes
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    def _make_layer(self, block, stride=1):
        pass
    def forward(self, x):
        return x

if __name__ == "__main__":
    x = torch.randn((1,64,224,224), dtype=torch.float)
    net = ResNet(ResidualBlock, [3,3,6,3])
    print(net(x).shape)