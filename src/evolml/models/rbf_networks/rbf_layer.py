import torch
from torch import nn

class RBFLayer(nn.Module):
    def __init__(self, input_size, out_size, centers=None, widths=None, device='cpu'):
        super().__init__()

        if centers is None:
            centers = torch.nn.init.uniform_(torch.zeros((input_size, out_size), device=device))
        elif not isinstance(centers, torch.Tensor):
            centers = torch.Tensor(centers, device=device)
 
        if widths is None:
            widths = torch.nn.init.uniform_(torch.ones((out_size,), device=device))
        elif not isinstance(widths, torch.Tensor):
            widths = torch.Tensor(widths, device=device)

        self.centers = torch.nn.Parameter(centers)
        self.widths = torch.nn.Parameter(widths)

    def forward(self, x):
        x = x.view(x.shape + (1,))
        centers = self.centers.view((1,) + self.centers.shape)
        widths = self.widths.view((1,) + self.widths.shape)

        return torch.exp(-torch.square(x - centers).sum(axis=1) * self.widths)
