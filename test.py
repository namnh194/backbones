import torch
import torch.nn as nn


class myBatchNorm2d(nn.Module):
    def __init__(self, input_size=None, epsilon=1e-3, momentum=0.99):
        super(myBatchNorm2d, self).__init__()
        assert input_size, print('Missing input_size parameter.')

        # Batch mean & var must be defined during training
        self.mu = torch.zeros(1, input_size[0], input_size[1], input_size[2])
        self.var = torch.ones(1, input_size[0], input_size[1], input_size[2])

        # For numerical stability
        self.epsilon = epsilon

        # Exponential moving average for mu & var update 
        self.it_call = 0  # training iterations
        self.momentum = momentum  # EMA smoothing

        # Trainable parameters
        self.beta = torch.nn.Parameter(torch.zeros(1, input_size[0], input_size[1], input_size[2]))
        self.gamma = torch.nn.Parameter(torch.ones(1, input_size[0], input_size[1], input_size[2]))

        # Batch size on which the normalization is computed
        self.batch_size = 0

    def forward(self, x):
        # [batch_size, input_size]

        self.it_call += 1

        if (self.batch_size == 0):
            # First iteration : save batch_size
            self.batch_size = x.shape[0]

        # Training : compute BN pass
        batch_mu = (x.sum(dim=0) / x.shape[0]).unsqueeze(0)  # [1, input_size]
        print("batch_mu size:", batch_mu.shape)
        batch_var = (x.var(dim=0) / x.shape[0]).unsqueeze(0)  # [1, input_size]
        print("batch_var size:", batch_var.shape)
        print("batch_var value:", batch_var)

        x_normalized = (x - batch_mu) / torch.sqrt(batch_var + self.epsilon)  # [batch_size, input_size]
        print("x_norm size:", x_normalized.shape)
        print("x_norm value:", x_normalized)
        x_bn = self.gamma * x_normalized + self.beta  # [batch_size, input_size]
        print("x_bn size:", x_bn.shape)
        print("x_bn value:", x_bn)

        # Update mu & std
        if (x.shape[0] == self.batch_size):
            running_mu = batch_mu
            running_var = batch_var
        else:
            running_mu = batch_mu * self.batch_size / x.shape[0]
            running_var = batch_var * self.batch_size / x.shape[0]

        self.mu = running_mu * (self.momentum / self.it_call) + \
                  self.mu * (1 - (self.momentum / self.it_call))
        self.var = running_var * (self.momentum / self.it_call) + \
                   self.var * (1 - (self.momentum / self.it_call))
        return x_bn  # [batch_size, output_size=input_size]

if __name__ == "__main__":
    net = myBatchNorm2d(input_size=(3,1,1))
    x = torch.randn((1,3,1,1), dtype=torch.float)
    print("output shape: ",net(x).shape)
