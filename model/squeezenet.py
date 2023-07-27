import torch
import torch.nn as nn
import torch.nn.functional as F
class fire_module(nn.Module):
    def __init__(self, in_channels, s1x1, e1x1, e3x3, is_conv=False):
        super(fire_module, self).__init__()
        self.expand = 2
        self.is_conv = is_conv
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, s1x1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(s1x1),
            nn.ReLU())
        self.expand1 = nn.Sequential(
            nn.Conv2d(s1x1, e1x1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(e1x1))
        self.expand2 = nn.Sequential(
            nn.Conv2d(s1x1, e3x3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(e3x3))
        self.relu = nn.ReLU()
        self.conv = None
        if self.is_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, e3x3+e1x1, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(e3x3+e1x1))
    def forward(self, x):
        residual = x
        out = self.squeeze(x)
        out = torch.cat([self.expand1(out), self.expand2(out)], dim=1)
        if self.is_conv:
            residual = self.conv(x)
        out += residual
        out = self.relu(out)
        return out

class SqueezeNet(nn.Module):
    def __init__(self, block, num_classes=10):
        super(SqueezeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(96),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fire2 = block(96, 16, 64, 64, True)
        self.fire3 = block(128, 16, 64, 64, False)
        self.fire4 = block(128, 32, 128, 128, True)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fire5 = block(256, 32, 128, 128, False)
        self.fire6 = block(256, 48, 192, 192, True)
        self.fire7 = block(384, 48, 192, 192, False)
        self.fire8 = block(384, 64, 256, 256, True)
        self.pool8 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fire9 = block(512, 64, 256, 256, False)
        self.dropout = nn.Dropout(0.5)
        self.conv10 = nn.Sequential(
            nn.Conv2d(512, 1000, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1000),
            nn.ReLU())
        self.pool10 = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(1000, num_classes)
    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.fire2(out)
        out = self.fire3(out)
        out = self.fire4(out)
        out = self.pool4(out)
        out = self.fire5(out)
        out = self.fire6(out)
        out = self.fire7(out)
        out = self.fire8(out)
        out = self.pool8(out)
        out = self.fire9(out)
        out = self.dropout(out)
        out = self.conv10(out)
        out = self.pool10(out)
        out = self.fc(out.view(out.shape[0], -1))
        return out


if __name__ == "__main__":
    net = SqueezeNet(fire_module, num_classes=10)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    x = torch.randn((1,3,224,224), dtype=torch.float)
    print(net(x).shape)