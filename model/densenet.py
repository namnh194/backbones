import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, times):
        super(ConvBlock, self).__init__()
        self.times = times
        self.block = self._make_block(in_channels, out_channels, times)
    def conv(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        return conv
    def _make_block(self, in_channels, out_channels, times):
        layer = []
        layer.append(self.conv(in_channels=in_channels, out_channels=out_channels))
        for i in range(times):
            layer.append(self.conv(in_channels=out_channels, out_channels=out_channels))
        return nn.Sequential(*layer)
    def forward(self, x):
        x = self.block(x)
        return x

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()
        self.transition = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
    def forward(self, x):
        out = self.transition(x)
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, times):
        super(DenseBlock, self).__init__()
        self.times = times
        self.in_channels = in_channels
        self.out_channels = out_channels
    def forward(self, x):
        x1 = ConvBlock(self.in_channels, self.out_channels, self.times[0])(x)
        _ = torch.cat([x,x1], dim=1)
        x2 = ConvBlock(_.shape[1], self.out_channels, self.times[1])(_)
        _ = torch.cat([x,x1,x2], dim=1)
        x3 = ConvBlock(_.shape[1], self.out_channels, self.times[2])(_)
        _ = torch.cat([x,x1,x2,x3], dim=1)
        x4 = ConvBlock(_.shape[1], self.out_channels, self.times[3])(_)
        del _
        _ = torch.cat([x,x1,x2,x3,x4], dim=1)
        return _

class DenseNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dense1 = DenseBlock(out_channels, out_channels, [6,6,6,6])
        self.transition1 = TransitionBlock(out_channels=out_channels)

    def forward(self, x):
        return

if __name__ == "__main__":
    net = DenseBlock(in_channels=3, out_channels=64, times=[3,2,3])
    x = torch.randn((1,3,24,24), dtype=torch.float)
    print(net(x).shape)