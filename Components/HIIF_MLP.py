import torch.nn as nn

class HIIF_MLP(nn.Module):
    def __init__(
            self, 
            in_dim, 
            out_dim, 
            hidden_dim, 
            act_layer=nn.GELU, 
            drop=0.0
        ):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        short_cut = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if x.shape[-1] == short_cut.shape[-1]:
            x = x + short_cut
        return x