import math

import numpy as np
import torch
from torch import nn

WALUDI = [1, 2, 4, 5, 6, 22, 23, 33, 64, 75, 84, 89, 96, 98, 101]
ISSC_indian= [0, 5, 7, 21, 25, 27, 29, 34, 37, 38, 41, 66, 70, 74, 83, 90, 97, 99, 100, 112, 121, 122, 128, 131, 135, 136, 137, 141, 148, 153, 164, 167, 176, 193]
LP_indian=[55, 56, 58, 77, 85, 88, 94, 96, 97, 100, 104, 105, 106, 107, 108, 112, 113, 116, 117, 118, 120, 124, 129, 131, 133, 140, 144, 148, 150, 151, 152, 153, 154, 159]
PREDEFINED_MASK = None

ALGO_NAME = 'WALUMI'
def create_boolean_tensor(vector, size):
    shifted_vector = np.array(vector)-1
    tensor = torch.zeros(size, dtype=torch.bool)
    tensor[shifted_vector] = True
    return tensor

class FeatureSelector(nn.Module):
    def __init__(self, input_dim, sigma, device, headstart_idx = None):
        super(FeatureSelector, self).__init__()
        self.input_dim = input_dim
        self.headstart_idx_to_tensor(headstart_idx)
        self.mu = torch.nn.Parameter(0.01 * torch.randn(input_dim, ), requires_grad=True)
        self.noise = torch.randn(self.mu.size())
        self.sigma = sigma
        self.device = device
        self.mask = None
        self.const_masking = None
        if PREDEFINED_MASK is not None:
            self.const_masking = create_boolean_tensor(PREDEFINED_MASK, input_dim)

    def apply_ndim_mask(self, mask_1d: torch.Tensor, x: torch.Tensor):
        mask = mask_1d.view(1, 1, -1, 1, 1).expand(*x.shape)
        #mask = np.tile(mask_1d[np.newaxis,np.newaxis,:, np.newaxis, np.newaxis], x.shape)
        return x.to(self.device) * mask.to(self.device)

    def apply_mask_loop(self, mask_1d: torch.Tensor, x: torch.Tensor):
        #batch,bands,p,p <- x.shape
        #batch p,p,bands -> x.shape
        x = x.squeeze()
        x = torch.transpose(x, 1, 3)
        x = x*mask_1d
        x = torch.transpose(x, 1, 3)
        batch_size, bands, p, p = x.shape
        #for idx in range(x.shape[0]):
        #    x[idx] = (x[idx].T*mask_1d).T
        return x

    def forward(self, x):
        z = self.mu + self.sigma * self.noise.normal_() * self.training
        stochastic_gate = self.hard_sigmoid(z)
        x = x.squeeze()
        x = torch.transpose(x, 1, 3)
        x = x * stochastic_gate
        x = torch.transpose(x, 1, 3)
        batch_size, bands, p, p = x.shape
        # for idx in range(x.shape[0]):
        #    x[idx] = (x[idx].T*mask_1d).T
        return x.unsqueeze(1)
        # For testing constant bands list of external method
        if self.const_masking is not None:
            with torch.no_grad():
                if len(prev_x.shape) == 5:
                    return prev_x.to(self.device) * self.const_masking.view(1, 1, self.const_masking.shape[0], 1, 1).to(
                        self.device)
                return prev_x.to(self.device) * self.const_masking.to(self.device)

        if self.mask is None:
            z = self.mu + self.sigma * self.noise.normal_() * self.training
            stochastic_gate = self.hard_sigmoid(z)
        else:
            stochastic_gate = self.mask

        return self.apply_mask_loop(stochastic_gate, prev_x).unsqueeze(1)
        # maybe in here expand the mask to (103,3,3)?
        if len(prev_x.shape) == 4:
            #return stochastic_gate.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(self.device) * prev_x.to(self.device)
            return prev_x.to(self.device) * stochastic_gate.view(1,1,stochastic_gate.shape[0],1).to(self.device)
        if len(prev_x.shape) == 5:
            #return prev_x.to(self.device) * stochastic_gate.view(1, 1, stochastic_gate.shape[0], 1, 1).to(self.device)
            #return stochastic_gate.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(self.device) * prev_x.to(self.device)
            return self.apply_ndim_mask(mask_1d=stochastic_gate,x=prev_x)
        return prev_x * stochastic_gate


    def hard_sigmoid(self, x):
        return torch.clamp(x + 0.5, 0.0, 1.0)

    def regularizer(self, x):
        #if self.const_masking is not None:
        #    return torch.Tensor([0])
        ''' Gaussian CDF. '''
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    def _apply(self, fn):
        super(FeatureSelector, self)._apply(fn)
        self.noise = fn(self.noise)
        return self

    def set_mask(self,mask):
        self.mask = mask

    def get_gates(self, mode):
        if mode == 'raw':
            return self.mu.detach().cpu().numpy()
        elif mode == 'prob':
            return np.minimum(1.0, np.maximum(0.0, self.mu.detach().cpu().numpy() + 0.5))
        else:
            raise NotImplementedError()
    def headstart_idx_to_tensor(self, headstart_idx):
        return
        if headstart_idx is not None:
            print(sorted(headstart_idx))
            headstart = torch.zeros(self.input_dim)
            self.headstart_idx = sorted(headstart_idx)
            shifted_vector = np.array(headstart_idx) - 1
            headstart[shifted_vector] = 1
            self.mu = torch.nn.Parameter(0.01*(torch.randn(self.input_dim, ))+0.05*headstart, requires_grad=True)
            self.noise = torch.randn(self.mu.size())
        else:
            self.mu = torch.nn.Parameter(0.01 * torch.randn(self.input_dim, ), requires_grad=True)
            self.noise = torch.randn(self.mu.size())
