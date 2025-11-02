import torch.nn as nn

class DecoderBase(nn.Module):
    def FeatureExtractor(self, inp):
        pass
    
    def Query(self, coord, cell):
        pass

    def forward(self, inp, coord, cell):
        pass