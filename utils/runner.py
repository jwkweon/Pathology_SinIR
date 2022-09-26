import time
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import save_image

from utils.augmentation import *
from utils.losses import *
from model.network import *

class Runner:
    def __init__(self, img_loader, save_dir, opt):
        self.img_loader = img_loader(opt, GeoAugment, Augment)
        self.save_dir = save_dir
        self.opt = opt

        self.loss = Loss(self.opt.losses)
        self.scale_num = self.opt.scale_num

        self.niter = self.opt.niter
        self.pixel_shuffle_p = self.opt.pixel_shuffle_p

        self.train_loader = self.img_loader.train_dataloader

        self.net_list = [Network(self.opt.nc_im, self.opt.nfc).to(self.opt.device)]

        self.lr = self.opt.lr
        self.betas = [self.opt.beta1, self.opt.beta2]
        self.optim = Adam(self.net_list[-1].parameters(), self.lr, self.betas)
        self.optim_sch = CosineAnnealingLR(self.optim, self.niter)


    def save(self):
        #torch.save()
        pass

    def load(self):
        pass

    def _grow_network(self):
        #self.net
        pass

    def _calc_loss(self, out, ori_img):
        return self.loss(out, ori_img.to(self.opt.device))

    def _step(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.optim_sch.step()

    def _forward(self, img, mask, scale, iter, infer=False):
        x = img.to(self.opt.device)
        m = mask.to(self.opt.device)
        z = torch.randn(x.shape, device=self.opt.device)
        alpha = iter / self.niter
        x = alpha*0.8*z + ((1-alpha)*0.2+0.8)*x

        net = self.net_list[scale]
        y = net(x)

        return y
    
    def train(self):
        start_time = time.time()
        start_time_inloop = time.time()

        for scale in range(self.scale_num):
            pbar = tqdm(range(1, self.niter + 1))
            for iter_cnt in pbar:
                gt, img, mask = next(self.train_loader)

                out = self._forward(img, mask, scale, iter_cnt)
                loss = self._calc_loss(out, gt)
                self._step(loss)

                pbar.set_postfix(loss=loss)
            