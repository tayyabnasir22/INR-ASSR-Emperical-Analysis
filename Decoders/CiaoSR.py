from Components.CiaoSR_CrossScaleAttention import CiaoSR_CrossScaleAttention
from Components.LIIF_MLP import LIIF_MLP
from Decoders.DecoderBase import DecoderBase
from Utilities.CoordinateManager import CoordinateManager
import torch
import torch.nn.functional as F

class CiaoSR(DecoderBase):

    def __init__(self, encoder):
        super().__init__()
        self.softmax_scale = 1
        self.local_size=2
        multi_scale=[4] 
        self.encoder = encoder
        imnet_q = {'in_dim':4,
            'out_dim':3,
            'hidden_list':[256, 256, 256, 256]}
        imnet_k = {'in_dim':64,
            'out_dim':64,
            'hidden_list':[256, 256, 256, 256]}
        imnet_v = {'in_dim':64,
            'out_dim':64,
            'hidden_list':[256, 256, 256, 256]}
        
        imnet_dim = self.encoder.out_dim

        imnet_q['in_dim'] = imnet_dim * 9
        imnet_k['in_dim'] = imnet_k['out_dim'] = imnet_dim * 9
        imnet_v['in_dim'] = imnet_v['out_dim'] = imnet_dim * 9

        imnet_k['in_dim'] += 4
        imnet_v['in_dim'] += 4
        
        imnet_q['in_dim'] += imnet_dim*len(multi_scale)
        imnet_v['in_dim'] += imnet_dim*len(multi_scale)
        imnet_v['out_dim'] += imnet_dim*len(multi_scale)

        self.imnet_q = LIIF_MLP(imnet_q['in_dim'], imnet_q['out_dim'], imnet_q['hidden_list']) 
        self.imnet_k = LIIF_MLP(imnet_k['in_dim'], imnet_k['out_dim'], imnet_k['hidden_list']) 
        self.imnet_v = LIIF_MLP(imnet_v['in_dim'], imnet_v['out_dim'], imnet_v['hidden_list']) 

        self.cs_attn = CiaoSR_CrossScaleAttention(channel=imnet_dim, scale=multi_scale)


    def FeatureExtractor(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        return self.feat
    
    def Query(self, coord, cell):
        
        feat = self.feat
        
        B, C, H, W = feat.shape

        _, h, w, _ = coord.shape
        
        res_con = F.grid_sample(
            self.inp, 
            coord.flip(-1), 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=False
        )

        coord = coord.view(B, h*w, 2)


        feat_q = F.unfold(feat, 3, padding=1).view(B, C*9, H, W)         #[16, 576, 48, 48]
        feat_k = F.unfold(feat, 3, padding=1).view(B, C*9, H, W)         #[16, 576, 48, 48]
    
        non_local_feat_v = self.cs_attn(feat)                        #[16, 64, 48, 48]
        feat_v = F.unfold(feat, 3, padding=1).view(B, C*9, H, W)     #[16, 576, 48, 48]
        feat_v = torch.cat([feat_v, non_local_feat_v], dim=1) 


        query = F.grid_sample(
            feat_q, 
            coord.flip(-1).unsqueeze(1), 
            mode='nearest', 
            align_corners=False
        ).permute(0, 3, 2, 1).contiguous()       #[16, 2304, 1, 576]

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
        preds_k, preds_v = [], []
        bs, q = coord.shape[:2]
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()  # [16, 2304, 2]
                coord_[:, :, 0] += vx /abs(vx) * rx + eps_shift  # [16, 2304]
                coord_[:, :, 1] += vy /abs(vy) * ry + eps_shift  # [16, 2304]
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                # key and value
                key = F.grid_sample(
                    feat_k, 
                    coord_.flip(-1).unsqueeze(1), 
                    mode='nearest', 
                    align_corners=False
                )[:, :, 0, :].permute(0, 2, 1).contiguous()       #[16, 2304, 576]
                value = F.grid_sample(
                    feat_v, 
                    coord_.flip(-1).unsqueeze(1), 
                    mode='nearest', 
                    align_corners=False
                )[:, :, 0, :].permute(0, 2, 1).contiguous()       #[16, 2304, 576]

                #Interpolate K to HR resolution
                coord_k = F.grid_sample(
                    pos_lr, 
                    coord_.flip(-1).unsqueeze(1), 
                    mode='nearest', 
                    align_corners=False
                )[:, :, 0, :].permute(0, 2, 1).contiguous()     #[16, 2304, 2]
                
                Q, K = coord, coord_k   #[16, 2304, 2]

                rel_coord = Q - K             #[16, 2304, 2]
                rel_coord[:, :, 0] *= feat.shape[-2]   # without mul
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = rel_coord   #[16, 2304, 2]

                rel_cell = cell.clone()      #[16, 2304, 2]

                rel_cell = rel_cell.unsqueeze(-2).repeat(1, coord.shape[1], 1)

                rel_cell[:, 0] *= feat.shape[-2]
                rel_cell[:, 1] *= feat.shape[-1]

                inp_v = torch.cat([value, inp, rel_cell], dim=-1)   #[16, 2304, 580]

                inp_k = torch.cat([key, inp, rel_cell], dim=-1)     #[16, 2304, 580]

                inp_k = inp_k.contiguous().view(bs * q, -1)
                inp_v = inp_v.contiguous().view(bs * q, -1)

                weight_k = self.imnet_k(inp_k).view(bs, q, -1).contiguous()   #[16, 2304, 576]
                
                pred_k = (key * weight_k).view(bs, q, -1)              #[16, 2304, 576]
                
                weight_v = self.imnet_v(inp_v).view(bs, q, -1).contiguous()   #[16, 2304, 576]

                
                pred_v = (value * weight_v).view(bs, q, -1)            #[16, 2304, 576]

                preds_v.append(pred_v)
                preds_k.append(pred_k)


        preds_k = torch.stack(preds_k, dim=-1)                      # [16, 2304, 576, 4]
        preds_v = torch.stack(preds_v, dim=-2)                      # [16, 2304, 4, 576]

        attn = (query @ preds_k)                                    # [16, 2304, 1, 4]
        x = ((attn/self.softmax_scale).softmax(dim=-1) @ preds_v)   # [16, 2304, 1, 576]

        result = self.imnet_q(x)               # [16, 2304, 3]

        result = result.view(bs, q, -1)

        result = result.view(bs, h, w, 3).permute(0, 3, 1, 2)    # [B, C, H, W]


        ret = result + res_con

        return ret

    def forward(self, inp, coord, cell):
        self.FeatureExtractor(inp)
        return self.Query(coord, cell)