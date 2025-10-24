from piq.psnr import psnr
from piq.ssim import ssim
from piq.gmsd import gmsd
from piq.vif import vif_p
from piq.fsim import fsim
from piq.srsim import srsim

class ScoreEvaluations:
    def __init__(self, pred, gt, lpips_model):
        self.psnr = psnr(pred, gt).item()
        self.ssim = ssim(pred, gt).item()
        self.gmsd = gmsd(pred, gt).item()
        self.vif = vif_p(pred, gt).item()
        self.fsim = fsim(pred, gt).item() if pred.shape[1] == 3 else 0.0
        self.sr_sim = srsim(pred, gt).item()
        self.lpips = lpips_model(pred, gt).item() if pred.shape[1] == 3 else 0.0
