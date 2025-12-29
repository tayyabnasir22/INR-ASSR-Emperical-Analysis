import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import init

class NaiveLinear(nn.Module):
    def __init__(self, features=3):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(features))

        self._weight = nn.Parameter(torch.empty(features, features))
        stdv = 1.0 / np.sqrt(8)
        init.uniform_(self._weight, -stdv, stdv)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        # y = x * A.t + b
        outputs = F.linear(inputs, self._weight, self.bias)
        logabsdet = self.get_logabsdet(self._weight)
        logabsdet = logabsdet * outputs.new_ones(batch_size)
        return outputs, logabsdet

    def inverse(self, inputs):
        outputs = inputs - self.bias
        outputs = torch.linalg.solve(self._weight, outputs.t())  # Linear-system solver.
        outputs = outputs.t()
        return outputs

    def get_logabsdet(self, x):
        # Note: torch.logdet() only works for positive determinant.
        _, res = torch.slogdet(x)
        return res

    def logabsdet(self):
        return self.get_logabsdet(self._weight)