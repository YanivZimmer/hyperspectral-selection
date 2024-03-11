import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
class FeatureSelector(nn.Module):
    def __init__(self, input_dim, sigma, device,target_number=1, temp=0.01):
        super(FeatureSelector, self).__init__()
        self.target_number = target_number
        self.device = device
        self.input_dim = input_dim
        self.headstart_idx = sorted(headstart_idx) if headstart_idx is not None else None
        self.mu = torch.nn.Parameter(
            0.01
            * torch.randn(
                input_dim,
                device=self.device
            ),
            requires_grad=True,
        )


    def forward(self, x):
        z = F.gumbel_softmax(self.mu, tau=self.temp, hard=True)
        # multiple the one hot\categorail aprrox
        # sum all values
        x = x.squeeze()
        x = torch.transpose(x, 1, 3)
        x = x * z
        #sum instead of argmax because argmax is not differential
        x = torch.sum(x, axis=3)
        x = torch.transpose(x, 1, 3)
        return x.unsqueeze(1)