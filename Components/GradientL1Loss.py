from Utilities.ImageProcessor import ImageProcessor
import torch.nn as nn

class GradientL1Loss(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_grad=0.5):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_grad = lambda_grad
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        # 1. Pixel L1 loss
        l1_loss = self.l1(pred, target)

        # 2. Gradient L1 loss (first-order derivatives)
        gx_pred, gy_pred = ImageProcessor.FirstOrderDerivativeSobel(pred)
        gx_tgt, gy_tgt = ImageProcessor.FirstOrderDerivativeSobel(target)

        grad_loss = self.l1(gx_pred, gx_tgt) + self.l1(gy_pred, gy_tgt)

        # 3. Weighted sum
        return self.lambda_l1 * l1_loss + self.lambda_grad * grad_loss