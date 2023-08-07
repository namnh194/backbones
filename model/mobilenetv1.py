import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

class depthwise(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(depthwise, self).__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, groups=in_channels, padding=1, stride=stride),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    def forward(self, x):
        out = self.dw(x)
        return out

class mobilenetv1(nn.Module):
    def __init__(self, num_classes=10):
        super(mobilenetv1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.conv2 = depthwise(32, 64, 1)
        self.conv3 = depthwise(64, 128, 2)
        self.conv4 = depthwise(128, 128, 1)
        self.conv5 = depthwise(128, 256, 2)
        self.conv6 = depthwise(256, 256, 1)
        self.conv7 = depthwise(256, 512, 2)
        self.conv8 = nn.ModuleList([depthwise(512, 512, 1)] * 5)
        self.conv9 = depthwise(512, 1024, 2)
        self.conv10 = depthwise(1024, 1024, 1)
        self.pool11 = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Sequential(
            nn.Linear(1024, 1000),
            nn.Dropout(0.4),
            nn.Linear(1000, num_classes))
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        for layer in self.conv8:
            out = layer(out)
        out = self.conv9(out)
        out = self.conv10(out)
        out = self.pool11(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    net = mobilenetv1()
    x = torch.randn((1,3,224,224), dtype=torch.float)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    print(net(x).shape)