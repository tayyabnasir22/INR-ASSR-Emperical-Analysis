from Configurations.BenchmarkType import BenchmarkType
from Configurations.NormalizerRange import NormalizerRange
from Models.RunningAverage import RunningAverage
from Models.ScoreEvaluations import ScoreEvaluations
from Utilities.CoordinateManager import CoordinateManager
from Utilities.Evaluation import Evalutaion
import random
import math
from functools import partial
from tqdm import tqdm
from Utilities.ImageProcessor import ImageProcessor
import torch

class PredictionHelpers:
    @staticmethod
    def PredictForBatch(model, inp, coord, cell, bsize):
        with torch.no_grad():
            model.gen_feat(inp)
            n = coord.shape[1]
            ql = 0
            preds = []
            while ql < n:
                qr = min(ql + bsize, n)
                pred = model.query_rgb(coord[:, ql: qr, :], cell)
                preds.append(pred)
                ql = qr
            pred = torch.cat(preds, dim=2)
        return pred
    
    @staticmethod
    def EvaluateForTrainigData(data_loader, model, input_nomrlizer_range: NormalizerRange, eval_bsize: int, scale: int = 4, dataset = BenchmarkType.DIV2K):
        model.eval()

        inp_sub = torch.FloatTensor(input_nomrlizer_range.sub).view(1, -1, 1, 1).cuda()
        inp_div = torch.FloatTensor(input_nomrlizer_range.div).view(1, -1, 1, 1).cuda()
        gt_sub = torch.FloatTensor(input_nomrlizer_range.sub).view(1, 1, -1).cuda()
        gt_div = torch.FloatTensor(input_nomrlizer_range.div).view(1, 1, -1).cuda()

        psnr_res = RunningAverage()
        ssim_res = RunningAverage()

        pbar = tqdm(data_loader, leave=False, desc='val')
        for batch in pbar:
            for k, v in batch.items():
                batch[k] = v.cuda(non_blocking=True)

            inp = (batch['inp'] - inp_sub) / inp_div

            coord = batch['coord']
            cell = batch['cell']

            pred = PredictionHelpers.PredictForBatch(model, inp, coord, cell * max(scale, 1), eval_bsize)
                
            pred = pred * gt_div + gt_sub
            pred.clamp_(0, 1)

            # Process the images for evaluation, shaving the required pixels
            sr, hr = ImageProcessor.PreprocessingForScoring(pred, batch['gt'], dataset, scale=scale)

            # For each metric in the list calculate the value
            psnr = Evalutaion.PSNRTrain(sr, hr)
            ssim = Evalutaion.SSIMTrain(sr, hr)
            psnr_res.add(psnr.item(), inp.shape[0])
            ssim_res.add(ssim.item(), inp.shape[0])

        return {
            'psnr': psnr_res.item(), 'ssim': ssim_res.item()
        }
    
    @staticmethod
    def EvaluteForTesting(data_loader, model, lpips_model, input_nomrlizer_range: NormalizerRange, eval_bsize: int, scale: int = 4, dataset = BenchmarkType.DIV2K):
        model.eval()

        inp_sub = torch.FloatTensor(input_nomrlizer_range.sub).view(1, -1, 1, 1).cuda()
        inp_div = torch.FloatTensor(input_nomrlizer_range.div).view(1, -1, 1, 1).cuda()
        gt_sub = torch.FloatTensor(input_nomrlizer_range.sub).view(1, 1, -1).cuda()
        gt_div = torch.FloatTensor(input_nomrlizer_range.div).view(1, 1, -1).cuda()

        val_res = {
            'psnr' : RunningAverage(), 
            'ssim' : RunningAverage(), 
            'gmsd' : RunningAverage(), 
            'fsim': RunningAverage(),
            'vif' : RunningAverage(), 
            'sr_sim' : RunningAverage(), 
            'lpips' : RunningAverage(), 
        }

        pbar = tqdm(data_loader, leave=False, desc='val')
        for batch in pbar:
            for k, v in batch.items():
                batch[k] = v.cuda(non_blocking=True)

            inp = (batch['inp'] - inp_sub) / inp_div

            coord = batch['coord']
            cell = batch['cell']

            pred = PredictionHelpers.PredictForBatch(model, inp, coord, cell * max(scale, 1), eval_bsize)
                
            pred = pred * gt_div + gt_sub
            pred.clamp_(0, 1)

            # Process the images for evaluation, shaving the required pixels
            sr, hr = ImageProcessor.PreprocessingForScoring(pred, batch['gt'], dataset, scale=scale)

            results = Evalutaion.GetEvaluationScores(sr, hr, lpips_model)

            for attr_name, value in vars(results).items():
                val_res[attr_name].add(value, 1)

            results = None

        return val_res