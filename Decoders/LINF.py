from Components.Flow import Flow
from Decoders.DecoderBase import DecoderBase
from Utilities.CoordinateManager import CoordinateManager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LINF(DecoderBase):
    def __init__(self, encoder, hidden_dim=256):
        super().__init__()

        num_layer = 3
        out_dim = 3
        flow_layers=10

        self.encoder = encoder
                
        self.coef = nn.Conv2d(encoder.out_dim, hidden_dim, out_dim, padding=1) # coefficient
        self.freq = nn.Conv2d(encoder.out_dim, hidden_dim, out_dim, padding=1) # frequency
        self.phase = nn.Linear(2, hidden_dim//2, bias=False) # phase 

        layers = []
        layers.append(nn.Conv2d(hidden_dim*4, hidden_dim, 1))
        layers.append(nn.ReLU())

        for i in range(num_layer-1):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, 1))
            layers.append(nn.ReLU())

        layers.append(nn.Conv2d(hidden_dim, flow_layers*out_dim*2, 1))
        self.layers = nn.Sequential(*layers)

        self.imnet = Flow(flow_layers=10)

    def FeatureExtractor(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        return self.feat

    def Query(self, coord, cell):
        feat = self.feat
        coef = self.coef(feat)
        freq = self.freq(feat)

        freqs = []
        coefs = []
        areas = []
        
        # prepare meta-data (coordinate)
        pos_lr = CoordinateManager.CreateCoordinates(
            feat.shape[-2:], 
            flatten=False
        ).cuda().permute(2, 0, 1).unsqueeze(0).expand(
            feat.shape[0], 
            2, 
            *feat.shape[-2:]
        )
        
        # local ensemble loop
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6
        
        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                # prepare coefficient & frequency
                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_coord = F.grid_sample(pos_lr, coord_.flip(-1), mode='nearest', align_corners=False)
                rel_coord = coord.permute(0, 3, 1, 2) - q_coord
                rel_coord[:, 0, :, :] *= feat.shape[-2]    # convert to relative scale i.e. each grid in the feature map has a range [-1, 1]
                rel_coord[:, 1, :, :] *= feat.shape[-1]

                # coefficient & frequency prediction
                coef_ = F.grid_sample(coef, coord_.flip(-1), mode='nearest', align_corners=False)
                freq_ = F.grid_sample(freq, coord_.flip(-1), mode='nearest', align_corners=False)
                
                # prepare cell
                rel_cell = cell.clone()
                rel_cell[:, 0] *= feat.shape[-2] # convert to relative scale i.e. each grid in the feature map has a range [-1, 1]
                rel_cell[:, 1] *= feat.shape[-1]

                # basis
                freq_ = torch.stack(torch.split(freq_, freq.shape[1]//2, dim=1), dim=2)
                freq_ = torch.mul(freq_, rel_coord.unsqueeze(1))
                freq_ = torch.sum(freq_, dim=2)
                freq_ += self.phase(rel_cell).unsqueeze(-1).unsqueeze(-1)
                freq_ = torch.cat((torch.cos(np.pi*freq_), torch.sin(np.pi*freq_)), dim=1)

                freqs.append(freq_)
                coefs.append(coef_)

                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)
        
        # apply local ensemble
        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t

        # weighted coefficeint & apply coefficeint to basis
        for i in range(4):
            w = (areas[i]/tot_area).unsqueeze(1)
            coefs[i] = torch.mul(w*coefs[i], freqs[i])

        # concat fourier features of 4 LR pixels
        features = torch.cat(coefs, dim=1)

        # shared MLP
        affine_info = self.layers(features)

        # flow
        bs, w, h, _ = coord.shape
        pred = self.imnet.inverse((torch.randn((bs * w * h, 3))), affine_info.permute(0, 2, 3, 1).contiguous().view(bs * w * h, -1))
        pred = pred.clone().view(bs, w, h, -1).permute(0, 3, 1, 2).contiguous()

        pred += F.grid_sample(self.inp, coord.flip(-1), mode='bilinear',\
                            padding_mode='border', align_corners=False)
        
        return pred

    def forward(self, inp, coord, cell):
        self.FeatureExtractor(inp)
        return self.Query(coord, cell)