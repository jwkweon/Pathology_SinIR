import os
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

import lpips
from utils.functions import norm, denorm, interp
from utils.ssim import SSIM
#from utils.interp_matlab import imresize


class Loss(nn.Module):
    def __init__(self, losses):
        super().__init__()
        self.loss_funcs = []

        for loss in losses:
            if 'ssim' in loss:
                win_size = int(loss[loss.rfind('m') + 1:])
                self.ssim = SSIM(data_range=1.0, win_size=win_size, nonnegative_ssim=True)
                self.loss_funcs.append(self._ssim)
            elif loss == 'mse': #L2
                self.mse = nn.MSELoss()
                self.loss_funcs.append(self._mse)
            elif loss == 'mae': #L1
                self.mae = nn.L1Loss()
                self.loss_funcs.append(self._mae)
            elif loss == 'lpips':
                self.lpips = lpips.LPIPS(net='vgg').cuda()
                self.loss_funcs.append(self._lpips)
            elif loss == 'crossentropy':
                self.ce = nn.CrossEntropyLoss()
                self.loss_funcs.append(self._ce)

    def _ssim(self, x, y):
        x, y = denorm(x), denorm(y)
        return 1 - self.ssim(x, y)

    def _mse(self, x, y):
        return self.mse(x, y)

    def _mae(self, x, y):
        return self.mae(x, y)

    def _lpips(self, x, y):
        return self.lpips(x, y)

    def _ce(self, x, y):
        return self.ce(x, y)

    def forward(self, x, y):
        loss = 0
        for loss_func in self.loss_funcs:
            loss = loss + loss_func(x, y)

        return loss


# def interp_matlab(x, img_shape):
#     if isinstance(img_shape, (tuple, list)):
#         return imresize(x, sides=img_shape, antialiasing=True)
#     elif isinstance(img_shape, int):
#         h, w = x.shape[-2:]
#         sf = img_shape / max(h, w)
#         h, w = int(h * sf), int(w * sf)
#         return imresize(x, sides=(h, w), antialiasing=True)
#     else:
#         raise Exception


def log_imgs(save_dir, imgs, desc):
    log_img = torch.cat([denorm(interp(img.detach(), imgs[-1].shape[-2:])) for img in imgs][::-1], dim=-2)

    os.makedirs(save_dir, exist_ok=True)
    path = f"{save_dir}/{desc}.png"

    save_image(log_img, path, padding=10)


def log(save_dir, iter_cnt, iter_num, out, start_time, start_time_inloop, scale_cur, scale_num):
    desc = f"[{scale_cur + 1}|{scale_num}][{iter_cnt}|{iter_num}]"
    log_imgs(f"{save_dir}/train_result", out, desc)

    time_log = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    time_log_inloop = time.strftime("%M:%S", time.gmtime(time.time() - start_time_inloop))
    logging.info(f"{time_log} | {time_log_inloop} | Iteration {desc}")
    start_time_inloop = time.time()


def shuffle_pixel(img, p=0.3):
    if p == 0:
        return img.clone()

    *_, h, w = img.shape
    out = img.clone()
    original_idx = torch.arange(h * w)
    shuffle_idx = torch.randperm(h * w)
    shuffle_idx = torch.where(torch.rand(*original_idx.shape) > p, original_idx, shuffle_idx)

    out = out.view(3, -1)[:, shuffle_idx].view(*out.shape)
    return out
