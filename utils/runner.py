import time
from tqdm import tqdm
import itertools

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder

from utils.augmentation import *
from utils.losses import *
from utils.functions import convert_image_np
from model.network import *

class Runner:
    def __init__(self, img_loader, save_dir, opt):
        self.img_loader = img_loader(opt)
        self.len_dataset = self.img_loader.len_ds

        self.save_dir = save_dir
        self.opt = opt

        # Define loss
        self.loss = Loss(self.opt.losses)   # default loss
        self.loss_lpips = Loss(['lpips'])  # ['mse', 'mae', 'lpips', 'ssim11', 'crossentropy']
        self.loss_ssim = Loss(['ssim11'])
        self.loss_ce = Loss(['crossentropy'])
        self.loss_mae = Loss(['mae'])
        self.loss_mse = Loss(['mse'])


        self.scale_num = self.opt.scale_num
        self.niter = self.opt.niter
        self.d_niter = self.opt.d_niter
        self.pixel_shuffle_p = self.opt.pixel_shuffle_p


        self.train_loader = self.img_loader.dataloader
        self.train_loader_iter = self.img_loader.dataloader_iter

        # Define network
        self.net_list = NetworkWithCode(self.opt.nc_im, self.opt.nfc).to(self.opt.device)
        self.d_net = SqueezeNet().to(self.opt.device)

        # Define Training Opt
        self.lr = self.opt.lr
        self.betas = [self.opt.beta1, self.opt.beta2]
        # self.optim = Adam(itertools.chain(self.net_list[0].parameters(), self.net_list[1].parameters()),
        #      self.lr, self.betas)

        self.optim_g = Adam(self.net_list.parameters(), self.lr, self.betas)
        self.optim_g_sch = CosineAnnealingLR(self.optim_g, self.niter)

        self.optim_d = Adam(self.d_net.parameters(), self.lr, self.betas)
        self.optim_d_sch = CosineAnnealingLR(self.optim_d, self.niter)


    def save(self):
        #torch.save()
        pass

    def load(self):
        pass

    def _grow_network(self):
        self.net_list.append(deepcopy(self.net_list[-1]))
        self.net_list[-2].requires_grad_(False)
        self.optim = Adam(self.net_list[-1].parameters(), self.lr, self.betas)
        self.optim_sch = CosineAnnealingLR(self.opt, self.iter_per_scale)

    def _set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def _calc_loss(self, out, ori_img):
        gt = ori_img.to(self.opt.device)
        return self.loss(out, gt)

    def _step(self, loss, optim, optim_sch):
        optim.zero_grad()
        loss.backward()
        optim.step()
        optim_sch.step()

    def _forward_d(self, img):
        x = img.to(self.opt.device)
        out_x = self.d_net(x)
        return out_x

    def _forward(self, img, code):
        x = img.to(self.opt.device)
        l = torch.eye(self.len_dataset)[code].to(self.opt.device)
        y = self.net_list(x, l)

        return y

    def train_d(self):
        self.d_net.train()
    
        pbar = tqdm(range(self.d_niter))
        for i in pbar:
            x, l = next(self.train_loader_iter)
            output = self._forward_d(x.to(self.opt.device))
            label = torch.eye(self.len_dataset)[l].to(self.opt.device)
            
            loss = self.loss_ce(output, label)
            self._step(loss, optim=self.optim_d, optim_sch=self.optim_d_sch)

            #pbar.set_postfix(loss=[loss.item()])
            pbar.set_postfix_str(f'loss={loss.item():.4f}')
        del pbar

    def train_self(self):
        #self.d_net.train()
        #self.optim_d = Adam(self.d_net.parameters(), self.lr, self.betas)
        #self.optim_d_sch = CosineAnnealingLR(self.optim_d, self.niter)

        aug = Augment()
        pbar = tqdm(range(1, self.niter + 1))
        for iter_cnt in pbar:
            img_b, l_b = next(self.train_loader_iter)
            # loss = 0
            if not img_b.is_cuda:
                img_b = img_b.to(self.opt.device)
                l_b = l_b.to(self.opt.device)

            for idx, (img,l) in enumerate(zip(img_b, l_b)):
                img_to_augment = convert_image_np(img)*255
                img = img.unsqueeze(0)
                data = {"image": img_to_augment}
                augmented = aug.transform(**data)
                augmented_img = augmented["image"]
            
                out = self._forward(augmented_img.unsqueeze(0), l)
                out_label = self.d_net(out)
                
                #print(out.shape, out_label, l, torch.eye(self.len_dataset)[[l]])
                loss_ce = self.loss_ce(out_label, torch.eye(self.len_dataset)[[l.unsqueeze(0)]].to(self.opt.device))
                #loss_ce = torch.tensor([0]).cuda()
                loss_ssim = self.loss_ssim(out, img)
                loss_mse = self.loss_mse(out, img)

                loss = loss_ssim + loss_mse
                #self._set_requires_grad(self.d_net, False)
                self._step(loss, optim=self.optim_g, optim_sch=self.optim_g_sch)

                #self._set_requires_grad(self.d_net, True)
                #self._step(loss_ce, optim=self.optim_d, optim_sch=self.optim_d_sch)

            pbar.set_postfix_str(f'TOT:{loss.item():.4f}, CE:{loss_ce.item():.4f}, SSIM:{loss_ssim.item():.4f}, MSE:{loss_mse.item():.4f}')
        del pbar, aug

    def train(self):
        start_time = time.time()
        start_time_inloop = time.time()

        pbar = tqdm(range(1, self.niter + 1))
        for iter_cnt in pbar:
            img, l = next(self.train_loader_iter)
            if not img.is_cuda:
                img = img.to(self.opt.device)
            if not l.is_cuda:
                l = l.to(self.opt.device)
                
            x_s, l_s = img[:self.opt.batch_size//2], l[:self.opt.batch_size//2]
            x_t, l_t = img[self.opt.batch_size//2:], l[self.opt.batch_size//2:]

            out_from_xs = self._forward(x_s, l_t)
            out_label = self.d_net(out_from_xs)

            rec_out = self._forward(out_from_xs, l_s)
            rec_label = self.d_net(rec_out)

            #print(x_s.is_cuda, l_s.is_cuda, x_t.is_cuda, l_t.is_cuda, out_from_xs.is_cuda, out_label.is_cuda, rec_out.is_cuda, rec_label.is_cuda)
            #loss = self.loss() torch.eye(self.len_dataset)[l_t].to(self.opt.device)
            loss_ce = self.loss_ce(out_label, torch.eye(self.len_dataset)[l_t].to(self.opt.device))
            loss_ce += self.loss_ce(rec_label, torch.eye(self.len_dataset)[l_s].to(self.opt.device))

            loss_rec = self.loss_mae(rec_out, x_s)
            loss_lpips = self.loss_lpips(x_s, out_from_xs)
            loss_lpips += self.loss_lpips(out_from_xs, rec_out)

            loss_ssim = self.loss_ssim(x_s, out_from_xs)
            loss_ssim += self.loss_ssim(out_from_xs, rec_out)

            loss = loss_rec + loss_ssim + loss_lpips.mean()
            # loss = loss_ce + loss_rec*10 + loss_lpips.mean() + loss_ssim

            self._step(loss, optim=self.optim_g, optim_sch=self.optim_g_sch)
            #self._step(loss_ce, optim=self.optim_d, optim_sch=self.optim_d_sch)
            
            #pbar.set_postfix(loss=[loss_cell.item(), loss_bg.item()])
            pbar.set_postfix_str(f'total_loss:{loss.item():.4f}, ce_loss:{loss_ce.item():.4f}, \
                rec_loss:{loss_rec.item():.4f}, lpips_loss:{loss_lpips.mean().item():.4f}')
        del pbar
