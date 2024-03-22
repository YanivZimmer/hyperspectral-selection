import torch
from torch import nn
import torch.nn.functional as F


class ConcreteEncoder(nn.Module):
    def __init__(self, input_dim, output_dim,device="cuda", start_temp=0.5, min_temp=0.01, alpha=0.99998):
        super().__init__()
        self.device = device
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.alpha = alpha

        self.temp = start_temp
        #out of input_dim select size of output_dim (emaple 25->1). multiple the 1 hot with the original data
        self.logits = nn.Parameter(torch.empty(output_dim, input_dim))
        nn.init.xavier_normal_(self.logits)
        self.regularizer = lambda : 0
        print(self.logits.detach().cpu().numpy())


    def forward(self, X, train=True, X_mask=None, debug=False):
        uniform = torch.rand(self.logits.shape).clamp(min=1e-7)
        gumbel = -torch.log(-torch.log(uniform)).to(self.device)*0.05
        self.temp = max([self.temp * self.alpha, self.min_temp])
        noisy_logits = (self.logits.to(self.device) + gumbel.to(self.device)) / self.temp

        if X_mask is not None:
            X *= X_mask
            logits_mask = X_mask.int() ^ 1
            noisy_logits = noisy_logits.reshape(1, self.logits.shape[0], -1)
            noisy_logits = torch.add(noisy_logits, logits_mask, alpha=-1e7)

        samples = F.softmax(noisy_logits, dim=-1)
        #print("samples",samples.shape,samples)
        discrete_logits = F.one_hot(torch.argmax(self.logits, dim=-1), self.logits.shape[1]).float()
        #print("discrete",discrete_logits.shape,discrete_logits)
        selection = samples if train else discrete_logits
        #print(X.shape,selection.shape,torch.transpose(selection, -1, -2).shape)
        Y = torch.matmul(X.transpose(2,4), torch.transpose(selection, -1, -2)).transpose(2,4)

        if debug:
            return X, selection
        #print("Y",Y.shape,Y)
        return Y

    def get_gates(self, mode):
        return self.logits.detach().cpu().numpy()
