from Models.RunningAverage import RunningAverage
from Utilities.CoordinateManager import CoordinateManager
from Utilities.Evaluation import Evalutaion
import random
import math
from functools import partial
from tqdm import tqdm
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
    def EvaluateForTrainigData(loader, model, data_norm=None, eval_type=None, eval_bsize=None, window_size=0, scale_max=4, fast=True,
              verbose=False):
        model.eval()

        if data_norm is None:
            data_norm = {
                'inp': {'sub': [0], 'div': [1]},
                'gt': {'sub': [0], 'div': [1]}
            }
        t = data_norm['inp']
        inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
        inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
        t = data_norm['gt']
        gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
        gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

        if eval_type is None:
            metric_fn = Evalutaion.PSNR
            metric_fn1 = Evalutaion.SSIM
        elif eval_type.startswith('div2k'):
            scale = int(eval_type.split('-')[1])
            metric_fn = partial(Evalutaion.PSNR, dataset='div2k', scale=scale)
            metric_fn1 = partial(Evalutaion.SSIM, dataset='div2k', scale=scale)
        elif eval_type.startswith('benchmark'):
            scale = int(eval_type.split('-')[1])
            metric_fn = partial(Evalutaion.PSNR, dataset='benchmark', scale=scale)
            metric_fn1 = partial(Evalutaion.SSIM, dataset='benchmark', scale=scale)
        else:
            raise NotImplementedError

        val_res = RunningAverage()
        ssim_res = RunningAverage()

        pbar = tqdm(loader, leave=False, desc='val')
        for batch in pbar:
            for k, v in batch.items():
                batch[k] = v.cuda(non_blocking=True)

            inp = (batch['inp'] - inp_sub) / inp_div

            # SwinIR Evaluation - reflection padding
            if window_size != 0:
                _, _, h_old, w_old = inp.size()
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                inp = torch.cat([inp, torch.flip(inp, [2])], 2)[:, :, :h_old + h_pad, :]
                inp = torch.cat([inp, torch.flip(inp, [3])], 3)[:, :, :, :w_old + w_pad]

                coord = CoordinateManager.CreateCoordinates((scale * (h_old + h_pad), scale * (w_old + w_pad)), flatten=False).unsqueeze(0).cuda()
                cell = torch.ones_like(batch['cell'])
                cell[:, 0] *= 2 / inp.shape[-2] / scale
                cell[:, 1] *= 2 / inp.shape[-1] / scale

            else:
                h_pad = 0
                w_pad = 0
                coord = batch['coord']
                cell = batch['cell']

            if eval_bsize is None:
                with torch.no_grad():
                    pred = model(inp, coord, cell)
            else:
                pred = PredictionHelpers.PredictForBatch(model, inp, coord, cell * max(scale / scale_max, 1),
                                        eval_bsize)
                
            pred = pred * gt_div + gt_sub
            pred.clamp_(0, 1)

            if eval_type is not None and window_size != 0:  # reshape for shaving-eval
                # gt reshape
                ih, iw = batch['inp'].shape[-2:]
                s = math.sqrt(batch['coord'].shape[1]*batch['coord'].shape[2] / (ih * iw))
                shape = [batch['inp'].shape[0], 3, round(ih * s), round(iw * s)]
                batch['gt'] = batch['gt'].view(*shape).contiguous()

                # prediction reshape
                ih += h_pad
                iw += w_pad
                s = math.sqrt(coord.shape[1]*coord.shape[2] / (ih * iw))
                shape = [batch['inp'].shape[0], 3, round(ih * s), round(iw * s)]
                pred = pred.view(*shape).contiguous()
                pred = pred[..., :batch['gt'].shape[-2], :batch['gt'].shape[-1]]

            res = metric_fn(pred, batch['gt'])
            res1 = metric_fn1(pred, batch['gt'])
            val_res.add(res.item(), inp.shape[0])
            ssim_res.add(res1.item(), inp.shape[0])

            if verbose:
                pbar.set_description('val {:.4f}'.format(val_res.item()))

        return val_res.item(), ssim_res.item()