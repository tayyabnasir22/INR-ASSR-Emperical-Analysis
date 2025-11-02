from Components.LIIF_MLP import LIIF_MLP
from Decoders.DecoderBase import DecoderBase
from Utilities.CoordinateManager import CoordinateManager
import torch
import torch.nn.functional as F
import numpy as np

class MetaSR(DecoderBase):

    def __init__(self, encoder, hidden_dim=256):
        super().__init__()
        self.encoder = encoder
        self.imnet = LIIF_MLP(3, self.encoder.out_dim * 9 * 3, [hidden_dim])
        

    def FeatureExtractor(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        return self.feat

    def Query(self, coord, cell):
        feat = self.feat

        feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        # prepare meta-data (coordinate)
        feat_coord = CoordinateManager.CreateCoordinates(feat.shape[-2:], flatten=False).cuda() \
                    .permute(2, 0, 1) \
                    .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        pos_lr = feat_coord.clone()
        pos_lr[:, :, 0] -= (2 / feat.shape[-2]) / 2
        pos_lr[:, :, 1] -= (2 / feat.shape[-1]) / 2

        B, H, W, _ = coord.shape
        coord_ = coord.clone()


        coord_ = coord_.view(B, H*W, 2)

        cell = cell.unsqueeze(-2).repeat(1, coord_.shape[1], 1)

        coord_[:, :, 0] -= cell[:, :, 0] / 2
        coord_[:, :, 1] -= cell[:, :, 1] / 2
        coord_q = (coord_ + 1e-6).clamp(-1 + 1e-6, 1 - 1e-6)
        q_feat = F.grid_sample(
            feat, coord_q.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_coord = F.grid_sample(
            feat_coord, coord_q.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)

        rel_coord = coord_ - q_coord
        rel_coord[:, :, 0] *= feat.shape[-2] / 2
        rel_coord[:, :, 1] *= feat.shape[-1] / 2

        r_rev = cell[:, :, 0] * (feat.shape[-2] / 2)
        inp = torch.cat([rel_coord, r_rev.unsqueeze(-1)], dim=-1)

        bs, q = coord_.shape[:2]
        pred = self.imnet(inp.view(bs * q, -1)).view(bs * q, feat.shape[1], 3)

        pred = torch.bmm(q_feat.contiguous().view(bs * q, 1, -1), pred)
        return pred.view(bs, H, W, 3).permute(0, 3, 1, 2)
    

    def forward(self, inp, coord, cell):
        self.FeatureExtractor(inp)
        return self.Query(coord, cell)
