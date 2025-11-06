from Utilities.ImageProcessor import ImageProcessor
import torch.nn as nn
import torch

class GramL1Loss(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_texture=0.5):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_texture = lambda_texture
        self.l1 = nn.L1Loss()

    def Gram(self, x):
        B, C, H, W = x.shape
        features = x.view(B, C, H*W)             # [B, C, HW]
        G = features @ features.transpose(1, 2)  # [B, C, C]
        return G / (C * H * W)                   # normalization (optional)

    def TextureLoss(self, pred, gt):
        G_pred = self.Gram(pred)
        G_gt = self.Gram(gt)
        return torch.mean((G_pred - G_gt) ** 2)

    def forward(self, pred, target):
        # 1. Pixel L1 loss
        l1_loss = self.l1(pred, target)

        # 2. Texture loss
        texture_loss = self.TextureLoss(pred, target)

        # 3. Weighted sum
        return self.lambda_l1 * l1_loss + self.lambda_texture * texture_loss