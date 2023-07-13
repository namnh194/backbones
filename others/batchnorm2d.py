import torch
import torch.nn as nn

class BatchNormalization(nn.Module):
    def __init__(self, num_channels=3):
        super(BatchNormalization, self).__init__()
        # param of destination distribution
        self.num_channels = num_channels
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))
        self.epsilon = 1e-5

    # x.shape: (batch_size, n_channels, height, weight)
    def forward(self, x):
        assert self.num_channels == x.shape[1]

        mean = torch.mean(x, dim=(0,2,3))
        variance = torch.var(x, dim=(0,2,3))

        for i in range(self.num_channels):
            # calculate x norm
            x[:,i,:,:] = (x[:,i,:,:] - mean[i]) / torch.sqrt(variance[i] - self.epsilon)
            # output of batch normalization
            x[:,i,:,:] = self.gamma[i] * x[:,i,:,:] + self.beta[i]
        return x

if __name__ == "__main__":
    x = torch.randn((8,3,224,224), dtype=torch.float)
    bn = BatchNormalization(num_channels=x.shape[1])
    bn_test = bn(x)
    bn_torch = nn.BatchNorm2d(x.shape[1])(x)

    assert torch.allclose(bn_test, bn_torch, rtol=0.02)