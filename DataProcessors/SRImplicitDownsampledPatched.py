from DataProcessors.SRDataProcessorBase import SRDataProcessorBase
from Utilities.CoordinateManager import CoordinateManager
from Utilities.ImageProcessor import ImageProcessor
import math
import torch
from PIL import Image

class SRImplicitDownsampledPatched(SRDataProcessorBase):
    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 patch_size=100, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        self.scale_max = scale_min if scale_max is None else scale_max
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = self.scale_min

        h_lr = math.floor(img.shape[-2] / s + 1e-9)
        w_lr = math.floor(img.shape[-1] / s + 1e-9)
        h_hr, w_hr = round(h_lr * s), round(w_lr * s)
        img = img[:, :h_hr, :w_hr]
        img_down = ImageProcessor.Resize(img, (h_lr, w_lr), Image.BICUBIC)
        crop_lr, crop_hr = img_down, img
        
        # --- Patch extraction (no overlap) ---
        lr_patches, hr_coord_patches, hr_rgb_patches, cell_patches = [], [], [], []
        for i in range(0, h_lr, self.patch_size):
            h_end_lr = min(i + self.patch_size, h_lr)
            hi, h_end_hr = i * s, h_end_lr * s

            for j in range(0, w_lr, self.patch_size):
                w_end_lr = min(j + self.patch_size, w_lr)
                hj, w_end_hr = j * s, w_end_lr * s

                # LR patch
                lr_patch = crop_lr[:, i:h_end_lr, j:w_end_lr]

                # HR patch (coords + rgb)
                hr_rgb_patch = crop_hr[:, int(hi):int(h_end_hr), int(hj):int(w_end_hr)]

                hr_coord_patch = CoordinateManager.CreateCoordinates([hr_rgb_patch.shape[-2], hr_rgb_patch.shape[-1]], flatten=False)

                # Cell per patch (normalized size)
                cell = torch.tensor([2 / hr_rgb_patch.shape[-2], 
                                     2 / hr_rgb_patch.shape[-1]], dtype=torch.float32)

                lr_patches.append(lr_patch.contiguous())
                hr_coord_patches.append(hr_coord_patch.contiguous())
                hr_rgb_patches.append(hr_rgb_patch.contiguous())
                cell_patches.append(cell)

        return {
            'inp_patches': lr_patches,         # list of [C, H_lr_patch, W_lr_patch]
            'coord_patches': hr_coord_patches, # list of [H_hr_patch, W_hr_patch, 2]
            'cell_patches': cell_patches,      # list of [2]
            'gt_patches': hr_rgb_patches,      # list of [C, H_hr_patch, W_hr_patch]
            'H': h_hr,
            'W': w_hr,
        }