import torch
import torch.nn as nn

class HIIF_Attention(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Linear(midc, midc * 3, bias=True)

        self.kln = nn.LayerNorm(self.headc)
        self.vln = nn.LayerNorm(self.headc)
        self.sm = nn.Softmax(dim=-1)

        self.proj1 = nn.Linear(midc, midc)
        self.proj2 = nn.Linear(midc, midc)

        self.proj_drop = nn.Dropout(0.)

        self.act = nn.GELU()

    def forward(self, x):
        B, HW, C = x.shape
        bias = x

        qkv = self.qkv_proj(x).reshape(B, HW, self.heads, 3 * self.headc)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1) # B, heads, HW, headc

        k = self.kln(k)
        v = self.vln(v)

        v = torch.matmul(k.transpose(-2, -1), v) / (HW)
        v = torch.matmul(q, v)
        v = v.permute(0, 2, 1, 3).reshape(B, HW, C)

        ret = v + bias
        bias = self.proj2(self.act(self.proj1(ret))) + bias

        return bias