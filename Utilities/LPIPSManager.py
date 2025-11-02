from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torch

class LPIPSManager:
    @staticmethod
    def GetBaseModelForLPIPS(net_type='squeeze'):
        lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type, normalize=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lpips_model = lpips.to(device)

        return lpips_model