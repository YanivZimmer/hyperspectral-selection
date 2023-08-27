import math

import numpy as np
import torch
from torch import nn

WALUDI = [1, 2, 4, 5, 6, 22, 23, 33, 64, 75, 84, 89, 96, 98, 101]
ISSC_indian = [
    0,
    5,
    7,
    21,
    25,
    27,
    29,
    34,
    37,
    38,
    41,
    66,
    70,
    74,
    83,
    90,
    97,
    99,
    100,
    112,
    121,
    122,
    128,
    131,
    135,
    136,
    137,
    141,
    148,
    153,
    164,
    167,
    176,
    193,
]
LP_indian = [
    55,
    56,
    58,
    77,
    85,
    88,
    94,
    96,
    97,
    100,
    104,
    105,
    106,
    107,
    108,
    112,
    113,
    116,
    117,
    118,
    120,
    124,
    129,
    131,
    133,
    140,
    144,
    148,
    150,
    151,
    152,
    153,
    154,
    159,
]
PREDEFINED_MASK = None

ALGO_NAME = "WALUMI"


def create_boolean_tensor(vector, size):
    shifted_vector = np.array(vector) - 1
    tensor = torch.zeros(size, dtype=torch.bool)
    tensor[shifted_vector] = True
    return tensor


class FeatureSelector(nn.Module):
    def __init__(self, input_dim, sigma, device, headstart_idx=None):
        super(FeatureSelector, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.headstart_idx_to_tensor(headstart_idx)
        self.mu = torch.nn.Parameter(
            0.01
            * torch.randn(
                input_dim,
            ),
            requires_grad=True,
        )
        self.noise = torch.randn(self.mu.size())
        self.sigma = sigma
        self.mask = None
        # if PREDEFINED_MASK is not None:
        #    self.const_masking = create_boolean_tensor(PREDEFINED_MASK, input_dim)

    def apply_ndim_mask(self, mask_1d: torch.Tensor, x: torch.Tensor):
        mask = mask_1d.view(1, 1, -1, 1, 1).expand(*x.shape)
        # mask = np.tile(mask_1d[np.newaxis,np.newaxis,:, np.newaxis, np.newaxis], x.shape)
        return x.to(self.device) * mask.to(self.device)

    def apply_mask_loop(self, mask_1d: torch.Tensor, x: torch.Tensor):
        # batch,bands,p,p <- x.shape
        # batch p,p,bands -> x.shape
        x = x.squeeze()
        x = torch.transpose(x, 1, 3)
        x = x * mask_1d
        x = torch.transpose(x, 1, 3)
        batch_size, bands, p, p = x.shape
        # for idx in range(x.shape[0]):
        #    x[idx] = (x[idx].T*mask_1d).T
        return x

    def forward(self, x):
        if self.mask is not None:
            return x * self.mask

        z = self.mu + self.sigma * self.noise.normal_() * self.training
        stochastic_gate = self.hard_sigmoid(z)
        if len(x.shape) == 2:
            return x * stochastic_gate
        x = x.squeeze()
        x = torch.transpose(x, 1, 3)
        x = x * stochastic_gate
        x = torch.transpose(x, 1, 3)
        return x.unsqueeze(1)

    def hard_sigmoid(self, x):
        return torch.clamp(x + 0.5, 0.0, 1.0)

    def regularizer(self, x):
        # if self.const_masking is not None:
        #    return torch.Tensor([0])
        """Gaussian CDF."""
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    def _apply(self, fn):
        super(FeatureSelector, self)._apply(fn)
        self.noise = fn(self.noise)
        return self

    def set_mask(self, mask):
        self.mask = mask

    def get_gates(self, mode):
        if mode == "raw":
            return self.mu.detach().cpu().numpy()
        elif mode == "prob":
            return np.minimum(
                1.0, np.maximum(0.0, self.mu.detach().cpu().numpy() + 0.5)
            )
        else:
            raise NotImplementedError()

    def headstart_idx_to_tensor(self, headstart_idx, const_mask=True):
        if headstart_idx is None:
            return
        print(sorted(headstart_idx))
        headstart_mask = torch.zeros(self.input_dim).to(self.device)
        self.headstart_idx = sorted(headstart_idx)
        shifted_vector = np.array(headstart_idx) - 1
        headstart_mask[shifted_vector] = 1
        if const_mask:
            self.const_masking = headstart_mask
            return
        self.mu = torch.nn.Parameter(
            0.01
            * (
                torch.randn(
                    self.input_dim,
                )
            )
            + 0.05 * headstart_mask,
            requires_grad=True,
        )
