import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    def forward(self, x):
        x = self.conv(x)
        return x

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, num_1x1, num_3x3, num_5x5, num_3x3_reduce, num_5x5_reduce, pooling):
        super(InceptionBlock, self).__init__()
        self.conv_1x1 = ConvBlock(in_channels, num_1x1, kernel_size=1, stride=1, padding=0)
        self.conv_3x3 = nn.Sequential(
            ConvBlock(in_channels, num_3x3_reduce, kernel_size=1, stride=1, padding=0),
            ConvBlock(num_3x3_reduce, num_3x3, kernel_size=3, stride=1, padding=1))
        self.conv_5x5 = nn.Sequential(
            ConvBlock(in_channels, num_5x5_reduce, kernel_size=1, stride=1, padding=0),
            ConvBlock(num_5x5_reduce, num_5x5, kernel_size=1, stride=1, padding=0))
        self.pooling = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, pooling, kernel_size=1, stride=1, padding=0))
    def forward(self, x):
        out_1x1 = self.conv_1x1(x)
        out_3x3 = self.conv_3x3(x)
        out_5x5 = self.conv_5x5(x)
        out_pool = self.pooling(x)
        out = torch.cat([out_1x1, out_3x3, out_5x5, out_pool], dim=1)
        return out

class InceptionV1(nn.Module):
    def __init__(self, input_channel, num_classes=10):
        super(InceptionV1, self).__init__()
        self.conv1 = ConvBlock(in_channels=input_channel, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Sequential(
            ConvBlock(64, 192, kernel_size=1, stride=1, padding=0),
            ConvBlock(192, 192, kernel_size=3, stride=1, padding=1))
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception_3a = InceptionBlock(192, 64, 128, 32, 96, 16, 32)
        self.inception_3b = InceptionBlock(256, 128, 192, 96, 128, 32, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception_4a = InceptionBlock(480, 192, 208, 48, 96, 16, 64)
        self.inception_4b = InceptionBlock(512, 160, 224, 64, 112, 24, 64)
        self.inception_4c = InceptionBlock(512, 128, 256, 64, 128, 24, 64)
        self.inception_4d = InceptionBlock(512, 112, 288, 64, 144, 32, 64)
        self.inception_4e = InceptionBlock(528, 256, 320, 128, 160, 32, 128)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception_5a = InceptionBlock(832, 256, 320, 128, 160, 32, 128)
        self.inception_5b = InceptionBlock(832, 384, 384, 128, 192, 48, 128)
        self.pool5 = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Sequential(nn.Linear(1024, 512), nn.Linear(512, num_classes))
    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = self.inception_3a(out)
        out = self.inception_3b(out)
        out = self.pool3(out)
        out = self.inception_4a(out)
        out = self.inception_4b(out)
        out = self.inception_4c(out)
        out = self.inception_4d(out)
        out = self.inception_4e(out)
        out = self.pool4(out)
        out = self.inception_5a(out)
        out = self.inception_5b(out)
        out = self.pool5(out)
        out = self.dropout(out)
        out = out.view(out.shape[0],-1)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    net = InceptionV1(input_channel=3, num_classes=10)
    x = torch.randn((1,3,224,224), dtype=torch.float)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    print(net(x).shape)
