import torch
import torch.nn as nn

class RDB_Conv(nn.Module):
    def __init__(self, in_channels, grow_rate, k_size=3):
        super(RDB_Conv, self).__init__()
        c_in = in_channels
        G  = grow_rate
        self.conv = nn.Sequential(*[
            nn.Conv2d(c_in, G, k_size, padding=(k_size-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)