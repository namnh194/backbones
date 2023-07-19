import torch
import torch.nn as nn
import torch.nn.functional as F


class Resnext_block(nn.Module):
    def __init__(self, in_channels, out_channels, expand=2, cardinality=32, stride=1, down_sample=None):
        super(Resnext_block, self).__init__()
        self.down_sample = down_sample
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, groups=cardinality, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels*expand, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels*expand),
            nn.ReLU())
    def forward(self, x):
        out = self.conv(x)
        if self.down_sample:
            x = self.down_sample(x)
        out += x
        return out

class Resnext50(nn.Module):
    def __init__(self, num_classes, block, block_config, width=4, expand=2, cardinality=32):
        super(Resnext50, self).__init__()
        self.width = width
        self.expand = expand
        self.cardinality = cardinality

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, block_config[0], 1)
        self.layer2 = self._make_layer(block, 256, block_config[1], 2)
        self.layer3 = self._make_layer(block, 512, block_config[2], 2)
        self.layer4 = self._make_layer(block, 1024, block_config[3], 2)
        self.pool2 = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.fc = nn.Linear(2048, num_classes)
    def _make_layer(self, block, in_channels, num_layer, stride_first):
        layer = []
        out_channels = self.cardinality * self.width
        down_sample = None
        if stride_first != 1 or in_channels != out_channels * self.expand:
            # this down_sample layer is important that kernel_size = 3 can increase num parameters twice
            down_sample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*self.expand, kernel_size=1, stride=stride_first, padding=0),
                nn.BatchNorm2d(out_channels*self.expand),
                nn.ReLU())
        layer.append(block(in_channels, out_channels, expand=self.expand, stride=stride_first, down_sample=down_sample))
        for i in range(num_layer-1):
            layer.append(block(out_channels*self.expand, out_channels, expand=self.expand, down_sample=None))
        self.width *= 2
        return nn.Sequential(*layer)
    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool2(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out


if __name__ == "__main__":
    x = torch.randn((1,3,224,224), dtype=torch.float)
    net = Resnext50(10, Resnext_block, [3,4,6,3])
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    print(net(x).shape)
