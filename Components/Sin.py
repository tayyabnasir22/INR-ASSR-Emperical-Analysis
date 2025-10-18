import torch
import torch.nn as nn

class Sin(nn.Module):
    def forward(self, input):
        return torch.sin(input)