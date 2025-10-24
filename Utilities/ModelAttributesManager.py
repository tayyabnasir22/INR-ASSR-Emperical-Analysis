from torch.optim import Adam, MultiStepLR, CosineAnnealingWarmRestarts
import numpy as np
import os
import torch

class ModelAttributesManager:
    @staticmethod
    def ComputeParameters(model, text=False):
        tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
        if text:
            if tot >= 1e6:
                return '{:.2f}M'.format(tot / 1e6)
            else:
                return '{:.2f}K'.format(tot / 1e3)
        else:
            return tot

    @staticmethod
    def CreateAdamOptimizer(param_list, state: dict, lr: float = 4.e-4, load_sd=False):
        optimizer = Adam(param_list, lr=lr)
        if load_sd:
            optimizer.load_state_dict(state)
        return optimizer
    
    @staticmethod
    def CreateMultiStepLRSchedular(optimizer, milestones: list[int], gamma: float, last_epoch: int = -1):
        return MultiStepLR(optimizer, milestones, gamma, last_epoch)
    
    @staticmethod
    def CreateSGDRSchedular(optimizer, t_zero: int = 50, t_mult: int = 1, lr_min: float = 2.e-6, last_epoch: int = -1):
        # TODO see how this can be managed
        optimizer.lr = 2.e-6
        return CosineAnnealingWarmRestarts(optimizer, t_zero, t_mult, lr_min, last_epoch)
    
    @staticmethod
    def SaveModel(model, optimizer, epoch: int, save_path: str, save_name):
        sv_file = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, save_name + '.pth'))