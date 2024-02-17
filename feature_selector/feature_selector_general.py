import math

import numpy as np
import torch
from torch import nn


def create_boolean_tensor(vector, size):
    shifted_vector = np.array(vector) - 1
    tensor = torch.zeros(size, dtype=torch.bool)
    tensor[shifted_vector] = True
    return tensor


class FeatureSelector(nn.Module):
    def __init__(self, input_dim, sigma, device, headstart_idx=None):
        super(FeatureSelector, self).__init__()
        self.target_number = 10
        self.device = device
        self.input_dim = input_dim
        self.headstart_idx_to_tensor(headstart_idx)
        self.headstart_idx=headstart_idx
        self.mu = torch.nn.Parameter(
            0.01
            * torch.randn(
                input_dim,
            ),
            requires_grad=True,
        ) if (self.mask is None) else None
        self.noise = torch.randn(self.mu.size()) if (self.mask is None) else torch.randn(input_dim)
        self.sigma = sigma
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

    def get_topk_stable(self,input_tensor,k):
        values, indices = torch.topk(input_tensor, k=k, largest=True)

        # Get the k-th largest value
        kth_value = values[-1]

        # Find the indices where the values are greater than the k-th value
        above_k_indices = torch.nonzero(input_tensor >= kth_value).squeeze()

        # Get the first k indices
        first_k_above_k_indices = above_k_indices[:k]
        first_k_above_k_indices = torch.sort(first_k_above_k_indices).values
        return first_k_above_k_indices

    def forward(self, x):
        discount = 1
        if self.headstart_idx is not None:
          x=x[:,:,self.headstart_idx]
          #print(x.shape)
          return x
        if self.mask is not None:
            if len(x.shape) == 2:
                return x * self.mask.to(x.device)
            x = x.squeeze()
            x = torch.transpose(x, 1, 3)
            x = x * self.mask.to(x.device)
            x = torch.transpose(x, 1, 3)
            return x.unsqueeze(1)

        z = self.mu + discount*self.sigma * self.noise.normal_() * self.training
        stochastic_gate = self.hard_sigmoid(z)
        if len(x.shape) == 2:
            return x * stochastic_gate
        x = x.squeeze()
        #k = int(0.05*x.shape[1])
        k = self.target_number# int(0.05 * stochastic_gate.shape[0])
        #topk = torch.topk(stochastic_gate, k,sorted = True).indices
        #topk = torch.sort(topk).values
        topk = self.get_topk_stable(stochastic_gate,k)
        x = x[:, topk]
        x = torch.transpose(x, 1, 3)
        x = x * stochastic_gate[topk]
        x = torch.transpose(x, 1, 3)
        return x.unsqueeze(1)

    def hard_sigmoid(self, x):
        return torch.clamp(x + 0.5, 0.0, 1.0)

    def reset_gates(self):
        self.mu = torch.nn.Parameter(
            0.01
            * torch.randn(
                self.input_dim,
                device=self.device
            ),
            requires_grad=True
        )
        self.noise = torch.randn(self.mu.size(),device="cuda")
        #self.mu = self.mu.to(self.device)
        #self.noise = self.noise.to(self.device)


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
        if self.mu is None:
            return None
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
            self.mask = None
            return
        print(sorted(headstart_idx))
        headstart_mask = torch.zeros(self.input_dim).to(self.device)
        self.headstart_idx = sorted(headstart_idx)
        shifted_vector = np.array(headstart_idx) #- 1
        headstart_mask[shifted_vector] = 1
        if const_mask:
            self.mask = headstart_mask
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
