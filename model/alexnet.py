import torch, torchvision
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets

class AlexNet(nn.Module):
    def __init__(self, n_channels=3):
        super(AlexNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )

    def forward(self, x):
        _ = self.block1(x)
        return _

if __name__ == "__main__":
    net = AlexNet()
    x = torch.rand((1,3,20,20), dtype=torch.float)
    print(net(x))