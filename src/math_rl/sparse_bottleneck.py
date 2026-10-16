"""Task-trained fixed-width features; sparsity ablation, not a CRATE replication."""
import torch
from torch import nn
from torch.nn import functional as F


class Bottleneck(nn.Module):
    def __init__(self, input_dim, width=768):
        super().__init__()
        self.projection = nn.Linear(input_dim, width)
        self.readout = nn.Linear(width, 1)

    def features(self, x):
        # Unit row norms prevent shrinking features while inflating readout weights
        # from trivially evading the activation penalty through projection scaling.
        return F.relu(F.linear(x, F.normalize(self.projection.weight, dim=1), self.projection.bias))

    def forward(self, x):
        z = self.features(x)
        return self.readout(z).flatten(), z


def loss(model, x, y, sparsity):
    logits, z = model(x)
    return F.binary_cross_entropy_with_logits(logits, y) + sparsity*z.mean()


@torch.no_grad()
def encode(model, x, mean, scale, device, batch=1024):
    model.eval()
    return torch.cat([model.features((v.to(device).float()-mean.to(device))/scale.to(device)).cpu()
                      for v in x.split(batch)])
