from Components.EDSRUpsampler import EDSRUpsampler
from Components.MeanShift import MeanShift
from Components.ResBlock import ResBlock
from Encoders.EncoderBase import EncoderBase
import torch.nn as nn

class EDSR(EncoderBase):
    def DefaultConv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), bias=bias)

    def __init__(self, n_resblocks=16, n_feats=64, res_scale=1,
                       scale=[2], no_upsampling=True, rgb_range=1, n_colors = 3):
        super(EDSR, self).__init__()
        self.no_upsampling = no_upsampling
        n_resblocks = n_resblocks
        n_feats = n_feats
        kernel_size = 3
        scale = scale[0]
        act = nn.ReLU(True)
        
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        # define head module
        m_head = [self.DefaultConv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                self.DefaultConv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(self.DefaultConv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        if self.no_upsampling:
            self.out_dim = n_feats
        else:
            self.out_dim = n_colors
            # define tail module
            m_tail = [
                EDSRUpsampler(self.DefaultConv, scale, n_feats, act=False),
                self.DefaultConv(n_feats, n_colors, kernel_size)
            ]
            self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        #x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        if self.no_upsampling:
            x = res
        else:
            x = self.tail(res)
        #x = self.add_mean(x)
        return x