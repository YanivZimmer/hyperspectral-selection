import math

import numpy as np
import torch
from torch import nn
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class FeatureSelectorL1Conv(nn.Module):

    def __init__(self,input_channels, threshold=1e-4):
        in_features, out_features = input_channels, input_channels
        super(FeatureSelectorL1Conv, self).__init__()
        self.threshold=threshold
        self.mu = torch.nn.Parameter(
            0.01
            * torch.ones(
                in_features,
            ),
            requires_grad=True,
        )

    def hard_sigmoid(self, x):
        return torch.clamp(x, 0.0, 1.0)

    def regularization(self):
        return torch.norm(self.mu, 1)

    def forward(self, x):
        #self.linear.weight.data *= self.mask
        #over_threshold = torch.abs(self.linear.weight.data) > self.threshold
        #self.linear.weight.data *= over_threshold
        #stochastic_gate = self.linear.weight.data
        #over_threshold = torch.abs(self.mu) > self.threshold
        #self.mu *= over_threshold
        temp = x
        stochastic_gate = self.hard_sigmoid(self.mu)
        print('above 0.0001',sum(stochastic_gate>0.0001))
        print('above 0.0',sum(stochastic_gate>0))
        print('below 0.009',sum(stochastic_gate<0.009))

        if len(x.shape) == 2:
            return x * stochastic_gate
        x = x.squeeze()

        x = torch.transpose(x, 1, 3)
        x = x * stochastic_gate
        x = torch.transpose(x, 1, 3)
        return x.unsqueeze(1)
