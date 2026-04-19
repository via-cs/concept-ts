import torch
import torch.nn as nn


class Rev_in(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.input_dim = input_dim
        gamma = torch.ones(1, 1, self.input_dim, requires_grad=True)
        self.gamma = nn.Parameter(gamma)

        beta = torch.zeros(1, 1, self.input_dim, requires_grad=True)
        self.beta = nn.Parameter(beta)

    def forward(self, x):
        batch_size = x.size(0)
        input_length = x.size(1)

        x_mean = x.mean(dim=1).reshape(batch_size, 1, self.input_dim).detach()
        x_std = torch.sqrt(
            x.var(dim=1, unbiased=True).reshape(batch_size, 1, self.input_dim) + 1e-5
        ).detach()

        self.x_mean = x_mean
        self.x_std = x_std

        x_mean = x_mean.expand(-1, input_length, -1)
        x_std = x_std.expand(-1, input_length, -1)
        x_gamma = self.gamma.expand(batch_size, input_length, -1)
        x_beta = self.beta.expand(batch_size, input_length, -1)

        x = x_gamma * ((x - x_mean) / x_std) + x_beta
        return x

    def de_rev_in(self, x):
        batch_size = x.size(0)
        input_length = x.size(1)

        x_mean = self.x_mean.expand(-1, input_length, -1)
        x_std = self.x_std.expand(-1, input_length, -1)
        x_gamma = self.gamma.expand(batch_size, input_length, -1)
        x_beta = self.beta.expand(batch_size, input_length, -1)

        output = x_std * ((x - x_beta) / x_gamma) + x_mean
        return output
