import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
x = torch.randn((1, 3, 224, 224), dtype=torch.float)
print(sum(p.numel() for p in shufflenet.parameters() if p.requires_grad))
print(shufflenet(x).shape)
layer = nn.GRU