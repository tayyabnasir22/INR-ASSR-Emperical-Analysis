from Components.HIIF_Attention import HIIF_Attention
from Components.HIIF_MLP import HIIF_MLP
from Decoders.DecoderBase import DecoderBase
from Utilities.CoordinateManager import CoordinateManager
import torch
import torch.nn as nn
import torch.nn.functional as F

class HIIF(DecoderBase):

    def __init__(self, encoder, blocks=16, hidden_dim=256):
        super().__init__()
        self.encoder = encoder
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)

        self.n_hi_layers = 6
        self.fc_layers = nn.ModuleList(
            [
                HIIF_MLP(
                    hidden_dim * 4 + 2 + 2 if d == 0 else hidden_dim + 2,
                    3 if d == self.n_hi_layers - 1 else hidden_dim,
                    256
                ) for d in range(self.n_hi_layers)
            ]
        )

        self.conv0 = HIIF_Attention(hidden_dim, blocks)
        self.conv1 = HIIF_Attention(hidden_dim, blocks)

    def FeatureExtractor(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        self.feat = self.freq(self.feat)
        return self.feat

    def Query(self, coord, cell):
        feat = (self.feat)
        grid = 0

        pos_lr = CoordinateManager.CreateCoordinates(
            feat.shape[-2:], 
            flatten=False
        ).cuda().permute(2, 0, 1).unsqueeze(0).expand(
            feat.shape[0], 
            2, 
            *feat.shape[-2:]
        )

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                feat_ = F.grid_sample(feat, coord_.flip(-1), mode='nearest', align_corners=False)

                old_coord = F.grid_sample(pos_lr, coord_.flip(-1), mode='nearest', align_corners=False)
                rel_coord = coord.permute(0, 3, 1, 2) - old_coord
                rel_coord[:, 0, :, :] *= feat.shape[-2] / 2
                rel_coord[:, 1, :, :] *= feat.shape[-1] / 2
                rel_coord_n = rel_coord.permute(0, 2, 3, 1).reshape(
                    rel_coord.shape[0], 
                    -1, 
                    rel_coord.shape[1]
                )

                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)

                preds.append(feat_)
                if vx == -1 and vy == -1:
                    # Local coord
                    rel_coord_mask = (rel_coord_n > 0).float()
                    rxry = torch.tensor([rx, ry], device=coord.device)[None, None, :]
                    local_coord = rel_coord_mask * rel_coord_n + (1. - rel_coord_mask) * (rxry - rel_coord_n)

        rel_cell = cell.clone()
        rel_cell[:, 0] *= feat.shape[-2]
        rel_cell[:, 1] *= feat.shape[-1]

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0];
        areas[0] = areas[3];
        areas[3] = t
        t = areas[1];
        areas[1] = areas[2];
        areas[2] = t

        for index, area in enumerate(areas):
            preds[index] = preds[index] * (area / tot_area).unsqueeze(1)

        grid = torch.cat(
            [
                *preds, 
                rel_cell.unsqueeze(-1).unsqueeze(-1).repeat(
                    1, 
                    1, 
                    coord.shape[1], 
                    coord.shape[2]
                )
            ], 
            dim=1
        )

        B, C_g, H, W = grid.shape
        grid = grid.permute(0, 2, 3, 1).reshape(B, H * W, C_g)

        for n in range(self.n_hi_layers):
            hi_coord = CoordinateManager.CreateHierarchicalCoordinates(local_coord, n)
            if n == 0:
                x = torch.cat([grid] + [hi_coord], dim=-1)
            else:
                x = torch.cat([x] + [hi_coord], dim=-1)
            x = self.fc_layers[n](x)
            if n == 0:
                x = self.conv0(x)
                x = self.conv1(x)

        result = x.permute(0, 2, 1).reshape(B, 3, H, W)

        ret = result + F.grid_sample(
            self.inp, 
            coord.flip(-1), 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=False
        )
        
        return ret

    def forward(self, inp, coord, cell):
        self.FeatureExtractor(inp)
        return self.Query(coord, cell)