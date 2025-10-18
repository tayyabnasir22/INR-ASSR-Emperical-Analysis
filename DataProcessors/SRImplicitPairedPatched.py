from Utilities.CoordinateManager import CoordinateManager
import torch

class SRImplicitPairedPatched:
    def __init__(self, dataset, inp_size=None, patch_size=100, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        h_lr, w_lr = img_lr.shape[-2:]
        h_hr = s * h_lr
        w_hr = s * w_lr
        img_hr = img_hr[:, :h_lr * s, :w_lr * s]
        crop_lr, crop_hr = img_lr, img_hr


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