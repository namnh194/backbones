import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, width=4, cardinality=32, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.cardinality = cardinality
        self.downsample = downsample
        self.conv = nn.ModuleList()
        for i in range(cardinality):
            self.conv.append(nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, out_channels*2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU()))
    def forward(self, x):
        feature = 0
        for i in range(self.cardinality):
            feature += self.conv[i](x)
        if self.downsample:
            x = self.downsample(x)
        feature = feature + x
        return feature

class ResNext50(nn.Module):
    def __init__(self, num_classes, block, config):
        super(ResNext50, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1 = self._make_layer(block, 64, 128, config[0], 1)
        self.block2 = self._make_layer(block, 256, 256, config[1], 2)
        self.block3 = self._make_layer(block, 512, 512, config[2], 2)
        self.block4 = self._make_layer(block, 1024, 1024, config[3], 2)
        self.pool2 = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes))
    def _make_layer(self, block, in_channels, out_channels, times, stride):
        layer = []
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels*2, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(in_channels*2),
                nn.ReLU())
        else:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels*4, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(in_channels*4),
                nn.ReLU())
        layer.append(block(in_channels, out_channels, stride=stride, downsample=downsample))
        for i in range(times-1):
            layer.append(block(out_channels*2, out_channels, stride=1, downsample=None))
        return nn.Sequential(*layer)
    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.pool2(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    x = torch.randn((1,3,224,224), dtype=torch.float)
    net = ResNext50(num_classes=10, block=ResidualBlock, config=[3,4,6,3])
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    print(net(x).shape)