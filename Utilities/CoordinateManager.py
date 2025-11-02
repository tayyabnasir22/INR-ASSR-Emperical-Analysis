import torch

class CoordinateManager:
    @staticmethod
    def CreateCoordinates(shape, ranges=None, flatten=True):
        coord_seqs = []
        for i, n in enumerate(shape):
            if ranges is None:
                v0, v1 = -1, 1
            else:
                v0, v1 = ranges[i]
            r = (v1 - v0) / (2 * n)
            seq = v0 + r + (2 * r) * torch.arange(n).float()
            coord_seqs.append(seq)
        ret = torch.stack(
            torch.meshgrid(
                *coord_seqs, 
                indexing="ij"
            ), 
            dim=-1
        )
        if flatten:
            ret = ret.view(-1, ret.shape[-1])
        return ret

    @staticmethod
    def GetRGBAndCoordiantes(img):
        coord = CoordinateManager.CreateCoordinates(img.shape[-2:])
        rgb = img.view(3, -1).permute(1, 0)
        return coord, rgb
    
    @staticmethod
    def CreateHierarchicalCoordinates(coord, n):
        coord_clip = torch.clip(coord - 1e-9, 0., 1.)
        coord_bin = ((coord_clip * 2 ** (n + 1)).floor() % 2)
        return coord_bin