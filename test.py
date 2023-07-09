from model.alexnet import *

model = AlexNet()

print(sum(p.numel() for p in model.parameters() if p.requires_grad))