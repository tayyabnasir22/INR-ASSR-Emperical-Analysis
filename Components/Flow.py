import torch.nn as nn
import torch
import numpy as np
from Components.NaiveLinear import NaiveLinear

class Flow(nn.Module):
    def __init__(self, flow_layers=10, patch_size=1, name='flow'):
        super(Flow, self).__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('device:', device)

        self.n_layers = flow_layers
        self.ps_square = patch_size*patch_size

        self.linears = torch.nn.ModuleList([NaiveLinear(3*patch_size*patch_size) for _ in range(flow_layers)])
        self.last = NaiveLinear(3*patch_size*patch_size)

        self.log2pi = float(np.log(2 * np.pi))
        self.affine_eps = 0.0001

    def get_logdet(self, scale):
        return torch.sum(torch.log(scale), dim=-1)

    def affine_forward(self, inputs, affine_info):
        scale, shift = torch.split(affine_info, 3*self.ps_square, dim=-1)
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        outputs = inputs * scale + shift
        logabsdet = self.get_logdet(scale)
        return outputs, logabsdet
        
    def affine_inverse(self, inputs, affine_info):
        scale, shift = torch.split(affine_info, 3*self.ps_square, dim=-1)
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        outputs = (inputs - shift) / scale
        return outputs

    def forward(self, x, affine_info):
        z, total_log_det_J = x, x.new_zeros(x.shape[0])
        for i in range(self.n_layers):
            z, log_det_J = self.linears[i](z)
            total_log_det_J += log_det_J
            z, log_det_J = self.affine_forward(z, affine_info[:, i*6*self.ps_square:i*6*self.ps_square+6*self.ps_square])
            total_log_det_J += log_det_J
        z, log_det_J = self.last(z)
        total_log_det_J += log_det_J
        # add base distribution log_prob
        total_log_det_J += torch.sum(-0.5 * (z ** 2 + self.log2pi), -1)
        return z, total_log_det_J
    
    def inverse(self, z, affine_info):
        x = z
        x = self.last.inverse(x)
        for i in reversed(range(self.n_layers)):
            x = self.affine_inverse(x, affine_info[:, i*6*self.ps_square:i*6*self.ps_square+6*self.ps_square])
            x = self.linears[i].inverse(x)
        return x