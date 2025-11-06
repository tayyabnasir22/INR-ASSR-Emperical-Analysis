from Models.BenchmarkType import BenchmarkType
from Models.NormalizerRange import NormalizerRange
from Models.RunningAverage import RunningAverage
from Utilities.Evaluation import Evalutaion
from tqdm import tqdm
from Utilities.ImageProcessor import ImageProcessor
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class PredictionHelpers:
    @staticmethod
    def InitNormalizers(input_nomrlizer_range: NormalizerRange):
        inp_sub = torch.FloatTensor(input_nomrlizer_range.sub).view(1, -1, 1, 1).cuda()
        inp_div = torch.FloatTensor(input_nomrlizer_range.div).view(1, -1, 1, 1).cuda()
        gt_sub = torch.FloatTensor(input_nomrlizer_range.sub).view(1, 1, -1).cuda()
        gt_div = torch.FloatTensor(input_nomrlizer_range.div).view(1, 1, -1).cuda()

        return inp_sub, inp_div, gt_sub, gt_div

    @staticmethod
    def InitEvaluator():
        return {
            'psnr' : RunningAverage(), 
            'ssim' : RunningAverage(), 
            'gmsd' : RunningAverage(), 
            'fsim': RunningAverage(),
            'vif' : RunningAverage(), 
            'sr_sim' : RunningAverage(), 
            'lpips' : RunningAverage(), 
        }


    @staticmethod
    def PredictForBatch(
        model: nn.Module, 
        inp: Tensor, 
        coord: Tensor, 
        cell: Tensor, 
        bsize: int
    ):
        with torch.no_grad():
            model.FeatureExtractor(inp)
            n = coord.shape[1]
            ql = 0
            preds = []
            while ql < n:
                qr = min(ql + bsize, n)
                pred = model.Query(coord[:, ql: qr, :], cell)
                preds.append(pred)
                ql = qr
            pred = torch.cat(preds, dim=2)
        return pred
    
    @staticmethod
    def GetNormalizedPrediction(
        model: nn.Module, 
        inp: Tensor, 
        coord: Tensor, 
        cell: Tensor, 
        scale: int, 
        eval_bsize: int, 
        gt_div: Tensor, 
        gt_sub: Tensor
    ):
        pred = PredictionHelpers.PredictForBatch(
            model, 
            inp, 
            coord, 
            cell * max(scale, 1), 
            eval_bsize
        )
            
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)
        return pred

    @staticmethod
    def EvaluateForTrainigData(
        data_loader: DataLoader, 
        model: nn.Module, 
        input_nomrlizer_range: NormalizerRange, 
        eval_bsize: int, 
        scale: int = 4, 
        dataset: BenchmarkType = BenchmarkType.DIV2K
    ):
        model.eval()

        inp_sub, inp_div, gt_sub, gt_div = PredictionHelpers.InitNormalizers(input_nomrlizer_range)

        psnr_res = RunningAverage()
        ssim_res = RunningAverage()

        pbar = tqdm(data_loader, leave=False, desc='val')
        for batch in pbar:
            for k, v in batch.items():
                batch[k] = v.cuda(non_blocking=True)

            inp = (batch['inp'] - inp_sub) / inp_div

            coord = batch['coord']
            cell = batch['cell']

            pred = PredictionHelpers.GetNormalizedPrediction(model, inp, coord, cell, scale, eval_bsize, gt_div, gt_sub)

            # Process the images for evaluation, shaving the required pixels
            sr, hr = ImageProcessor.PreprocessingForScoring(pred, batch['gt'], dataset, scale=scale)

            # For each metric in the list calculate the value
            psnr = Evalutaion.PSNRTrain(sr, hr)
            ssim = Evalutaion.SSIMTrain(sr, hr)
            psnr_res.SetItem(psnr.item(), inp.shape[0])
            ssim_res.SetItem(ssim.item(), inp.shape[0])

        return {
            'psnr': psnr_res.GetItem(), 'ssim': ssim_res.GetItem()
        }

    @staticmethod
    def Evaluate(
        pred: Tensor, 
        gt: Tensor, 
        dataset: BenchmarkType, 
        scale: int, 
        lpips_model: LearnedPerceptualImagePatchSimilarity, 
        val_res: dict
    ):# Process the images for evaluation, shaving the required pixels
        sr, hr = ImageProcessor.PreprocessingForScoring(
            pred, 
            gt, 
            dataset, 
            scale=scale
        )

        results = Evalutaion.GetEvaluationScores(sr, hr, lpips_model)

        for attr_name, value in vars(results).items():
            val_res[attr_name].SetItem(value, 1)

    @staticmethod
    def EvaluteForTesting(
        data_loader: DataLoader, 
        model: nn.Module, 
        lpips_model: LearnedPerceptualImagePatchSimilarity, 
        input_nomrlizer_range: NormalizerRange, 
        eval_bsize: int, 
        scale: int = 4, 
        dataset:BenchmarkType = BenchmarkType.DIV2K
    ):
        model.eval()

        inp_sub, inp_div, gt_sub, gt_div = PredictionHelpers.InitNormalizers(input_nomrlizer_range)

        val_res = PredictionHelpers.InitEvaluator()

        pbar = tqdm(data_loader, leave=False, desc='val')
        for batch in pbar:
            for k, v in batch.items():
                batch[k] = v.cuda(non_blocking=True)

            inp = (batch['inp'] - inp_sub) / inp_div

            coord = batch['coord']
            cell = batch['cell']

            pred = PredictionHelpers.GetNormalizedPrediction(model, inp, coord, cell, scale, eval_bsize, gt_div, gt_sub)

            PredictionHelpers.Evaluate(pred, batch['gt'], dataset, scale, lpips_model, val_res)

        return val_res
    
    @staticmethod
    def ProcessPatchBatch(
        batch: Tensor, 
        inp_sub: Tensor, 
        inp_div: Tensor, 
        gt_sub: Tensor, 
        gt_div: Tensor, 
        model: nn.Module, 
        eval_bsize: int, 
        scale: int
    ):
        gt_patches = []
        pred_patches = []

        for i in range(0, len(batch['inp_patches'])):
            for k, values in batch.items():
                if k not in ['H', 'W']:
                    batch[k] = [v.cuda(non_blocking=True) for v in values]

            inp = (batch['inp_patches'][i] - inp_sub) / inp_div

            coord = batch['coord_patches'][i]
            cell = batch['cell_patches'][i]

            pred = PredictionHelpers.GetNormalizedPrediction(model, inp, coord, cell, scale, eval_bsize, gt_div, gt_sub)

            pred_patches.append(pred)
            gt_patches.append(batch['gt_patches'][i])

        return gt_patches, pred_patches


    @staticmethod
    def EvaluteForPatchedTesting(
        data_loader: DataLoader, 
        model: nn.Module, 
        lpips_model: LearnedPerceptualImagePatchSimilarity, 
        input_nomrlizer_range: NormalizerRange, 
        eval_bsize: int, 
        patch_size: int,
        scale: int = 4, 
        dataset: BenchmarkType = BenchmarkType.DIV2K
    ):
        model.eval()

        inp_sub, inp_div, gt_sub, gt_div = PredictionHelpers.InitNormalizers(input_nomrlizer_range)

        val_res = PredictionHelpers.InitEvaluator()

        pbar = tqdm(data_loader, leave=False, desc='val')
        for batch in pbar:
            gt_patches, pred_patches = PredictionHelpers.ProcessPatchBatch(batch, inp_sub, inp_div, gt_sub, gt_div, model, eval_bsize, scale)

            # Merge the patches
            sr = ImageProcessor.MergePatches(
                patches=pred_patches, 
                H=int(batch['H'][0]),
                W=int(batch['W'][0]),
                patch_size=patch_size*scale
            )
            hr = ImageProcessor.MergePatches(
                patches=gt_patches, 
                H=int(batch['H'][0]),
                W=int(batch['W'][0]),
                patch_size=patch_size*scale
            )

            PredictionHelpers.Evaluate(sr, hr, dataset, scale, lpips_model, val_res)

        return val_res
    

    @staticmethod
    def EvaluteForOverlapPatchedTesting(
        data_loader: DataLoader, 
        model: nn.Module, 
        lpips_model: LearnedPerceptualImagePatchSimilarity, 
        input_nomrlizer_range: NormalizerRange, 
        eval_bsize: int, 
        patch_size: int,
        overlap: int,
        scale: int = 4, 
        dataset: BenchmarkType = BenchmarkType.DIV2K
    ):
        model.eval()

        inp_sub, inp_div, gt_sub, gt_div = PredictionHelpers.InitNormalizers(input_nomrlizer_range)

        val_res = PredictionHelpers.InitEvaluator()

        pbar = tqdm(data_loader, leave=False, desc='val')
        for batch in pbar:
            gt_patches, pred_patches = PredictionHelpers.ProcessPatchBatch(batch, inp_sub, inp_div, gt_sub, gt_div, model, eval_bsize, scale)

            # Merge the patches
            sr = ImageProcessor.MergePatchesOverlap(
                patches=pred_patches, 
                H=int(batch['H'][0]),
                W=int(batch['W'][0]),
                patch_size=patch_size*scale,
                overlap=overlap*scale
            )
            hr = ImageProcessor.MergePatchesOverlap(
                patches=gt_patches, 
                H=int(batch['H'][0]),
                W=int(batch['W'][0]),
                patch_size=patch_size*scale,
                overlap=overlap*scale
            )

            PredictionHelpers.Evaluate(sr, hr, dataset, scale, lpips_model, val_res)

        return val_res