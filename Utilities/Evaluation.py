from Models.ScoreEvaluations import ScoreEvaluations
from Utilities.ImageProcessor import ImageProcessor
from pytorch_msssim import ssim
import torch

class Evalutaion:
    @staticmethod
    def PSNR(sr, hr, dataset=None, scale=1, rgb_range=1):
        diff = (sr - hr) / rgb_range
        if dataset is not None:
            if dataset == 'benchmark':
                shave = scale
                if diff.size(1) > 1:
                    gray_coeffs = [65.738, 129.057, 25.064]
                    convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                    diff = diff.mul(convert).sum(dim=1)
            elif dataset == 'div2k':
                shave = scale + 6
            else:
                raise NotImplementedError
            valid = diff[..., shave:-shave, shave:-shave]
        else:
            valid = diff
        mse = valid.pow(2).mean()
        return -10 * torch.log10(mse)

    @staticmethod
    def SSIM(sr, hr, dataset=None, scale=1, rgb_range=1):
        sr, hr = ImageProcessor.PreprocessingForScoring(sr, hr, dataset, scale, rgb_range)
        return ssim(sr, hr, data_range=1.0, size_average=True)

    @staticmethod
    def GetEvaluationScores(sr, hr, dataset=None, scale=1, rgb_range=1):
        sr, hr = ImageProcessor.PreprocessingForScoring(sr, hr, dataset, scale, rgb_range)
        return ScoreEvaluations(sr, hr)