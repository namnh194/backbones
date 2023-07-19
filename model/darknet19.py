import torch
import torch.nn as nn
import torch.nn.functional as F


cfg = [32, 'pool', 64, 'pool', [128, 64, 128], 'pool', [256, 128, 256], 'pool', [512, 256, 512, 256, 512], 'pool', [1024, 512, 1024, 512, 1024]]
class Darknet19(nn.Module):
    def __init__(self, num_classes=10):
        global cfg
        super(Darknet19, self).__init__()
        model = []

        in_channels = 3
        for i, _ in enumerate(cfg):
            if isinstance(_, str):
                model += [nn.MaxPool2d(kernel_size=2, stride=2, padding=0)]
            elif isinstance(_, int):
                model += [self.conv(in_channels, _, kernel=3, stride=1, padding=1)]
                in_channels = _
            else:
                for j, __ in enumerate(_):
                    if j % 2 == 0:
                        model += [self.conv(in_channels, __, kernel=3, stride=1, padding=1)]
                    else:
                        model += [self.conv(in_channels, __, kernel=1, stride=1, padding=0)]
                    in_channels = __
        model += [self.conv(in_channels, 1000, kernel=1, stride=1, padding=0), nn.AdaptiveAvgPool2d(output_size=(1,1))]
        self.model = nn.Sequential(*model)
        self.fc = nn.Linear(1000, num_classes)
    def conv(self, in_channels, out_channels, kernel, stride, padding):
        conv = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels)]
        if kernel != 1:
            conv += [nn.ReLU()]
        return nn.Sequential(*conv)
    def forward(self, x):
        out = self.model(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    net = Darknet19(10)
    x = torch.randn((1,3,224,224), dtype=torch.float)
    out = net(x)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    print(out.shape)
    print(F.softmax(out, dim=1))