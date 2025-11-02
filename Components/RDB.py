from Components.RDB_Conv import RDB_Conv
import torch.nn as nn

class RDB(nn.Module):
    def __init__(
            self, 
            grow_rate_0, 
            grow_rate, 
            n_conv_layers,
        ):
        super(RDB, self).__init__()
        G0 = grow_rate_0
        G  = grow_rate
        C  = n_conv_layers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x