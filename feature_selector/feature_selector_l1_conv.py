import math

import numpy as np
import torch
from torch import nn
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class FeatureSelectorL1Conv(nn.Module):

    def __init__(self,input_channels, threshold=1e-4):
        in_features, out_features = input_channels, input_channels
        super(FeatureSelectorL1Conv, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False, device=device)
        #Mask used to make all non-diagonal matrix weights elements zero
        self.mask = torch.eye(in_features, dtype=bool, device=device)
        self.threshold = threshold

    def forward(self, x):
        self.linear.weight.data *= self.mask
        over_threshold = torch.abs(self.linear.weight.data) > self.threshold
        self.linear.weight.data *= over_threshold
        stochastic_gate = self.linear.weight.data
        if len(x.shape) == 2:
            return x * stochastic_gate
        x = x.squeeze()
        x = torch.transpose(x, 1, 3)
        x = x * stochastic_gate
        x = torch.transpose(x, 1, 3)
        return x.unsqueeze(1)
