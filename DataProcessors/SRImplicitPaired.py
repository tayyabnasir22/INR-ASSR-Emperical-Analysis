from DataProcessors.SRDataProcessorBase import SRDataProcessorBase
from Utilities.CoordinateManager import CoordinateManager
import random
import torch

class SRImplicitPaired(SRDataProcessorBase):
    def __init__(
            self, 
            dataset, 
            inp_size=None, 
            augment=False
        ):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            h_hr = s * h_lr
            w_hr = s * w_lr
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord = CoordinateManager.CreateCoordinates([h_hr, w_hr], flatten=False)
        hr_rgb = crop_hr
        
        if self.inp_size is not None:
            x0 = random.randint(0, h_hr - h_lr)
            y0 = random.randint(0, w_hr - w_lr)
            
            hr_coord = hr_coord[x0:x0+self.inp_size, y0:y0+self.inp_size, :]
            hr_rgb = crop_hr[:, x0:x0+self.inp_size, y0:y0+self.inp_size]
        
        cell = torch.tensor(
            [
                2 / crop_hr.shape[-2], 
                2 / crop_hr.shape[-1]
            ], 
            dtype=torch.float32
        )

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }