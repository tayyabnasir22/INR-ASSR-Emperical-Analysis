import torch.nn as nn

class DecoderBase(nn.Module):
    def gen_feat(self, inp):
        pass
    
    def query_rgb(self, coord, cell):
        pass

    def forward(self, inp, coord, cell):
        pass