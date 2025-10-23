from Models.ScoreEvaluations import ScoreEvaluations
from Utilities.ImageProcessor import ImageProcessor
from pytorch_msssim import ssim
import torch

class Evalutaion:
     # Note training eval metrics do not shave pixels off the edges
    @staticmethod
    def PSNRTrain(sr, hr, rgb_range=1):
        diff = (sr - hr) / rgb_range
        valid = diff
        mse = valid.pow(2).mean()
        return -10 * torch.log10(mse)

    @staticmethod
    def SSIMTrain(sr, hr, rgb_range=1):
        diff_sr = sr / rgb_range
        diff_hr = hr / rgb_range
        return ssim(diff_sr, diff_hr, data_range=1.0, size_average=True)

    @staticmethod
    def GetEvaluationScores(sr, hr, dataset=None, scale=1, rgb_range=1):
        sr, hr = ImageProcessor.PreprocessingForScoring(sr, hr, dataset, scale, rgb_range)
        return ScoreEvaluations(sr, hr)