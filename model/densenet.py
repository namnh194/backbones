import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, times):
        super(DenseBlock, self).__init__()
        self.times = times
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.layers = nn.ModuleList()
        for _ in range(times):
            self.layers.append(self._make_conv(in_channels, growth_rate))
            in_channels += growth_rate
    def _make_conv(self, in_channels, growth_rate):
        conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=growth_rate, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(),
            nn.Conv2d(in_channels=growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU())
        return conv
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            x = layer(x)
            features.append(x)
            x = torch.cat(features, dim=1)
        return x

class DenseNet(nn.Module):
    def __init__(self, num_classes, block_config, growth_rate=32):
        super(DenseNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=growth_rate*2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(growth_rate*2),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dense = nn.ModuleList()

        in_channels = growth_rate*2
        for i, num_layer in enumerate(block_config):
            block = DenseBlock(in_channels, growth_rate, num_layer)
            self.dense.add_module(f"dense_block {i+1}", block)
            in_channels += growth_rate * num_layer
            if i != len(block_config)-1:
                transition = self._make_transition(in_channels, in_channels//2)
                self.dense.add_module(f"transition_block {i+1}", transition)
                in_channels = in_channels//2
        self.pool2 = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.cls = nn.Linear(in_channels, num_classes)
    def _make_transition(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        for layer in self.dense:
            out = layer(out)
        out = self.pool2(out)
        out = out.view(out.shape[0], -1)
        out = self.cls(out)
        return out

if __name__ == "__main__":
    net = DenseNet(num_classes=10, block_config=[6,12,24,16], growth_rate=32)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    x = torch.randn((1,3,224,224), dtype=torch.float)
    print(net(x).shape)