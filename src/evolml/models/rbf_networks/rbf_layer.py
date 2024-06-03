import torch
from torch import nn

class RBFLayer(nn.Module):
    def __init__(self, input_size, out_size, device='cpu'):
        super().__init__()

        self.centers = torch.nn.Parameter(torch.nn.init.uniform_(torch.zeros((input_size, out_size), device=device)))
        self.inv_widths = torch.nn.Parameter(torch.nn.init.uniform_(torch.zeros((out_size,), device=device)))

    def forward(self, x):
        x = x.view(x.shape + (1,))
        centers = self.centers.view((1,) + self.centers.shape)
        inv_widths = self.inv_widths.view((1,) + self.inv_widths.shape)

        return torch.exp(torch.square(x - centers).sum(axis=1) * self.inv_widths)
