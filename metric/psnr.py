# ===============================================================
#
# Codes are obtained from https://github.com/bonlime/pytorch-tools
#
# ===============================================================
import torch

# Usage
'''
    from metric.psnr import PSNR
    psnr = PSNR(data_range=1.0)
    score = psnr(x,y)
'''

class PSNR:
    """Peak Signal to Noise Ratio
    X and Y have range [0, 255]"""

    def __init__(self,
                 data_range=255):
        self.name = "PSNR"
        self.data_range = data_range


    def __call__(self, X, Y):
        mse = torch.mean((X - Y) ** 2)
        return 20 * torch.log10(self.data_range / torch.sqrt(mse))