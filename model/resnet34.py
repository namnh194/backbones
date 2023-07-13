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
    def __init__(self, block, time_layers, num_classes=10):
        super(ResNet34, self).__init__()
        self.num_classes = num_classes
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer2 = self._make_layer(block, 64, time_layers[0], 1)
        self.layer3 = self._make_layer(block, 128, time_layers[1], 2)
        self.layer4 = self._make_layer(block, 256, time_layers[2], 2)
        self.layer5 = self._make_layer(block, 512, time_layers[3], 2)
        self.pool2 = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(512, num_classes)
    def _make_layer(self, block, kernel_size, times, stride_first=1):
        layer = []
        if stride_first != 1:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=kernel_size//2, out_channels=kernel_size, kernel_size=1, stride=stride_first),
                nn.BatchNorm2d(kernel_size))
            layer.append(block(in_channels=kernel_size//2, out_channels=kernel_size, stride=stride_first, downsample=downsample))
        else:
            layer.append(block(in_channels=kernel_size, out_channels=kernel_size))
        for i in range(1, times):
            layer.append(block(in_channels=kernel_size, out_channels=kernel_size))
        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.layer1(x)
        out = self.pool1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.pool2(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out


if __name__ == "__main__":
    x = torch.randn((1,3,227,227), dtype=torch.float)
    net = ResNet34(ResidualBlock, [3,4,6,3])
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    print(net(x).shape)
