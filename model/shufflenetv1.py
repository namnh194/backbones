import torch
import torch.nn as nn
import torch.nn.functional as F


class Shuffle_unit(nn.Module):
    def __init__(self, in_channels, out_channels, stride, group, down_sample=None):
        super(Shuffle_unit, self).__init__()
        self.group = group
        self.down_sample = down_sample
        self.g_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=group, stride=1, padding=0),
            nn.BatchNorm2d(out_channels))
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU()
    def shuffle_channel(self, x, group):
        batch_size, channel, height, width = x.shape
        channel_per_group = channel // group
        x = x.view(batch_size, group, channel_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch_size, -1, height, width)
        return x
    def forward(self, x):
        out = self.g_conv(x)
        out = self.shuffle_channel(out, self.group)
        out = self.conv(out)
        if self.down_sample:
            out = torch.cat([out, self.down_sample(x)], dim=1)
        else:
            out += x
        out = self.relu(out)
        return out

class ShuffleNet(nn.Module):
    def __init__(self, num_classes, block, group, output_group, block_config):
        super(ShuffleNet, self).__init__()
        self.group = group
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage2 = self._make_layer(block, 24, output_group[0], 2, block_config[0])
        self.stage3 = self._make_layer(block, output_group[0], output_group[1], 2, block_config[1])
        self.stage4 = self._make_layer(block, output_group[1], output_group[2], 2, block_config[2])
        self.pool2 = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Sequential(
            nn.Linear(output_group[-1], 1000),
            nn.Dropout(0.4),
            nn.Linear(1000, num_classes))
    def _make_layer(self, block, in_channels, out_channels, stride, num_layer):
        layer = []
        down_sample = None
        if stride != 1:
            down_sample = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        elif in_channels != out_channels:
            down_sample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels))
        layer += [block(in_channels, out_channels - in_channels, stride, self.group, down_sample)]
        for i in range(num_layer-1):
            layer += [block(out_channels, out_channels, stride=1, group=self.group, down_sample=None)]
        return nn.Sequential(*layer)
    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.pool2(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    group = 3
    if group == 1:
        output_group = [144, 288, 576]
    elif group == 2:
        output_group = [200, 400, 800]
    elif group == 3:
        output_group = [240, 480, 960]
    elif group == 4:
        output_group = [272, 544, 1088]
    else:
        output_group = [384, 768, 1536]

    x = torch.randn((1,3,224,224), dtype=torch.float)
    net = ShuffleNet(10, Shuffle_unit, group=2, output_group=output_group, block_config=[4, 8, 4])
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    print(net(x).shape)