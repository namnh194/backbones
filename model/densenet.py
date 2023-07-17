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


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, times):
        super(DenseBlock, self).__init__()
        self.times = times
        self.in_channels = in_channels
        self.out_channels = out_channels
    def forward(self, x):
        device = x.device
        x1 = ConvBlock(self.in_channels, self.out_channels, self.times[0])(x).to(device)
        _ = torch.cat([x,x1], dim=1).to(device)
        x2 = ConvBlock(_.shape[1], self.out_channels, self.times[1])(_).to(device)
        _ = torch.cat([x,x1,x2], dim=1).to(device)
        x3 = ConvBlock(_.shape[1], self.out_channels, self.times[2])(_).to(device)
        _ = torch.cat([x,x1,x2,x3], dim=1).to(device)
        x4 = ConvBlock(_.shape[1], self.out_channels, self.times[3])(_).to(device)
        _ = torch.cat([x,x1,x2,x3,x4], dim=1).to(device)
        del x1,x2,x3,x4
        return _

class DenseNet(nn.Module):
    def __init__(self, num_classes, in_channels, grown_rate=32):
        super(DenseNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=grown_rate*2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(grown_rate*2),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dense1 = DenseBlock(grown_rate*2, grown_rate*4, [6,6,6,6])
        self.transition1 = self._make_transition(grown_rate*18, grown_rate*9)
        self.dense2 = DenseBlock(grown_rate*9, grown_rate*6, [12,12,12,12])
        self.transition2 = self._make_transition(grown_rate*33, int(grown_rate*16.5))
        self.dense3 = DenseBlock(int(grown_rate*16.5), grown_rate*9, [24,32,48,64])
        self.transition3 = self._make_transition(int(grown_rate*52.5), int(grown_rate*4.5))
        self.dense4 = DenseBlock(int(grown_rate*4.5), int(grown_rate*12.5), [16,32,32,48])
        self.pool2 = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
        self.fc = nn.Linear(int(grown_rate*54.5)*7*7, 1000)
        self.cls = nn.Sequential(nn.Linear(1000,512), nn.Dropout(0.4), nn.Linear(512,num_classes))
    def _make_transition(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.dense1(out)
        out = self.transition1(out)
        out = self.dense2(out)
        out = self.transition2(out)
        out = self.dense3(out)
        out = self.transition3(out)
        out = self.dense4(out)
        out = self.pool2(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        out = self.cls(out)
        return out

if __name__ == "__main__":
    net = DenseNet(num_classes=10, in_channels=3, grown_rate=32)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    x = torch.randn((1,3,224,224), dtype=torch.float)
    print(net(x).shape)