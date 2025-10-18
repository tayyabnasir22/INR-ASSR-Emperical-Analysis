import torch
import torch.nn as nn

class Cos(nn.Module):
    def forward(self, input):
        return torch.cos(input)