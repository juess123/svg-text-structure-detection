import torch
import torch.nn as nn
import torch.nn.functional as F

class TextEnergyModel(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(feature_dim))
        self.raw_w = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, x):
        w = F.softplus(self.raw_w)
        diff = x - self.mu
        energy = torch.sum(w * diff * diff, dim=1)
        score = torch.exp(-energy)
        return score, energy