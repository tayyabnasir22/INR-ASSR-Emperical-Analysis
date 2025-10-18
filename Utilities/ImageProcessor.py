

from Utilities.CoordinateManager import CoordinateManager


class ImageProcessor:
    @staticmethod
    def PreprocessingForScoring(sr, hr, dataset=None, scale=1, rgb_range=1):
        # Normalize SR and HR images to [0,1] range (or scale defined by rgb_range)
        diff_sr = sr / rgb_range
        diff_hr = hr / rgb_range

        # If dataset is specified, apply dataset-specific preprocessing
        if dataset is not None:
            # For benchmark datasets (like Set5, Set14), 
            # crop 'scale' pixels from each border to avoid boundary effects
            if dataset == 'benchmark':
                shave = scale
                if sr.size(1) > 1: 
                    # If multi-channel (RGB) image
                    # Convert RGB to Y-channel (luminance) because most SR evaluation is done in Y channel of YCbCr space.
                    # Gray coefficients are from ITU-R BT.601 standard.
                    gray_coeffs = [65.738, 129.057, 25.064]
                    convert = sr.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256

                    # Weighted sum across RGB channels → single-channel luminance
                    diff_sr = diff_sr.mul(convert).sum(dim=1, keepdim=True)
                    diff_hr = diff_hr.mul(convert).sum(dim=1, keepdim=True)
            elif dataset == 'div2k':
                # For DIV2K dataset, crop 'scale+6' pixels from border
                # (DIV2K convention for fair evaluation)
                shave = scale + 6
            else:
                raise NotImplementedError

            # Apply cropping (shaving) to remove boundary pixels
            # [..., shave:-shave, shave:-shave] means crop H and W dimensions
            diff_sr = diff_sr[..., shave:-shave, shave:-shave]
            diff_hr = diff_hr[..., shave:-shave, shave:-shave]

        return diff_sr, diff_hr
    
    @staticmethod
    def Resize(img, size):
        return transforms.ToTensor()(
            transforms.Resize(size, InterpolationMode.BICUBIC)(
                transforms.ToPILImage()(img)))

    @staticmethod
    def SampleForModel(img, scale_min=1, scale_max=4, inp_size=None, augment=False, epoch=None):
        if epoch < 20: s = random.randint(scale_min, scale_max)
        s = random.uniform(scale_min, scale_max)
        # print(s)

        if inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            img = img[:, :, :h_hr, :w_hr]
            img_down = torch.stack([ImageProcessor.Resize(x, (h_lr, w_lr)) for x in img], dim=0)
            crop_lr, crop_hr = img_down, img
        else:
            h_lr = inp_size
            w_lr = inp_size
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, :, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = torch.stack([ImageProcessor.Resize(x, w_lr) for x in crop_hr], dim=0)

        if augment == True:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip: x = x.flip(-2)
                if vflip: x = x.flip(-1)
                if dflip: x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        coord = CoordinateManager.CreateCoordinates([h_hr, w_hr], flatten=False)
        coord = coord.unsqueeze(0).expand(img.shape[0], *coord.shape[:2], 2)

        cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32).unsqueeze(0).expand(
            img.shape[0], 2)
        return {
            'inp': crop_lr.contiguous(),
            'coord': coord.contiguous(),
            'cell': cell.contiguous(),
            'gt': crop_hr.contiguous()
        }
    
    @staticmethod
    def MergePatches(patches, H, W, patch_size=100):
        B, C = patches[0].shape[:2]
        merged = torch.zeros(B, C, H, W, device=patches[0].device, dtype=patches[0].dtype)

        idx = 0
        for i in range(0, H, patch_size):
            h_end = min(i + patch_size, H)
            for j in range(0, W, patch_size):
                w_end = min(j + patch_size, W)
                merged[:, :, i:h_end, j:w_end] = patches[idx]
                idx += 1

        return merged.contiguous()

    @staticmethod
    def MergePatchesOverlap(patches, H, W, patch_size=100, overlap=20):
        B, C = patches[0].shape[:2]
        merged = torch.zeros(B, C, H, W, device=patches[0].device, dtype=patches[0].dtype)
        weight = torch.zeros_like(merged)  # for blending overlaps

        stride = patch_size - overlap
        idx = 0
        for i in range(0, H, stride):
            h_end = min(i + patch_size, H)
            i = max(h_end - patch_size, 0)

            for j in range(0, W, stride):
                w_end = min(j + patch_size, W)
                j = max(w_end - patch_size, 0)

                patch = patches[idx]

                # blend: add and later normalize
                merged[:, :, i:h_end, j:w_end] += patch
                weight[:, :, i:h_end, j:w_end] += 1
                idx += 1

        merged /= weight  # average overlapping regions
        return merged.contiguous()

    @staticmethod
    def GetPatchesOverlap(lr_img, hr_coord, scale, patch_size=100, overlap=20):
        lr_patches = []
        hr_coord_patches = []
        B, C, H_lr, W_lr = lr_img.shape

        stride = patch_size - overlap
        for i in range(0, H_lr, stride):
            h_end = min(i + patch_size, H_lr)
            i = max(h_end - patch_size, 0)  # adjust so last patch fits

            for j in range(0, W_lr, stride):
                w_end = min(j + patch_size, W_lr)
                j = max(w_end - patch_size, 0)

                # LR patch
                lr_patch = lr_img[:, :, i:h_end, j:w_end]
                lr_patches.append(lr_patch.contiguous())

                # HR coords patch
                hi, hj = i * scale, j * scale
                h_end_hr, w_end_hr = h_end * scale, w_end * scale
                hr_patch = hr_coord[:, hi:h_end_hr, hj:w_end_hr, :]
                hr_coord_patches.append(hr_patch.contiguous())

        return lr_patches, hr_coord_patches

    @staticmethod
    def GetPatches(lr_img, hr_coord, scale, patch_size = 100):
        lr_patches = []
        hr_coord_patches = []
        B, C, H_lr, W_lr = lr_img.shape

        for i in range(0, H_lr, patch_size):
            h_end = min(i + patch_size, H_lr)
            for j in range(0, W_lr, patch_size):
                w_end = min(j + patch_size, W_lr)

                # LR patch
                lr_patch = lr_img[:, :, i:h_end, j:w_end]
                lr_patches.append(lr_patch.contiguous())

                # Corresponding HR coords patch
                hi, hj = i*scale, j*scale
                h_end_hr, w_end_hr = h_end*scale, w_end*scale
                hr_patch = hr_coord[:, hi:h_end_hr, hj:w_end_hr, :]
                hr_coord_patches.append(hr_patch.contiguous())

        return lr_patches, hr_coord_patches