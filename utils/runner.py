import time
from tqdm import tqdm
import itertools

import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder

from utils.augmentation import *
from utils.losses import *
from utils.functions import convert_image_np
from model.network import *

class Runner:
    def __init__(self, img_loader, save_dir, opt):
        self.opt = opt
        self.save_dir = save_dir

        self.img_loader = img_loader(opt)
        self.num_classes = self.opt.num_classes
        if self.opt.dir_test_path is not None:
            from utils.loader import TestLoader 
            self._loader = TestLoader(opt)
            self.test_loader = self._loader.dataloader
            self.test_loader_iter = self._loader.dataloader_iter

        # Define loss
        self.loss = Loss(self.opt.losses)   # default loss
        self.loss_lpips = Loss(['lpips'])  # ['mse', 'mae', 'lpips', 'ssim11', 'crossentropy']
        self.loss_ssim = Loss(['ssim7'])
        self.loss_ce = Loss(['crossentropy'])
        self.loss_mae = Loss(['mae'])
        self.loss_mse = Loss(['mse'])


        self.scale_num = self.opt.scale_num
        self.niter = self.opt.niter
        self.d_niter = self.opt.d_niter
        self.pixel_shuffle_p = self.opt.pixel_shuffle_p


        self.train_loader = self.img_loader.dataloader
        self.train_loader_iter = self.img_loader.dataloader_iter
        self.aug = JitAugment(policy='color,cutout,shuffle')    # translation decreases performance
        # self.rand_aug = RandAug()

        # Define network
        self.net_gen = NetworkWithCode_V2(self.opt.nc_im, self.opt.nfc, self.num_classes).to(self.opt.device)
        #self.net_rec = NetworkWithCode(self.opt.nc_im, self.opt.nfc).to(self.opt.device)
        self.d_net = SqueezeNet(self.num_classes).to(self.opt.device)

        # Define Training Opt
        self.lr = self.opt.lr
        self.betas = [self.opt.beta1, self.opt.beta2]
        # self.optim = Adam(itertools.chain(self.net_list[0].parameters(), self.net_list[1].parameters()),
        #      self.lr, self.betas)

        self.optim_g = Adam(self.net_gen.parameters(), self.lr, self.betas)
        # self.optim_g = SGD(self.net_gen.parameters(), self.lr, momentum=0.9)
        #self.optim_rec = Adam(self.net_rec.parameters(), self.lr, self.betas)
        self.optim_g_sch = CosineAnnealingLR(self.optim_g, self.niter)
        #self.optim_rec_sch = CosineAnnealingLR(self.optim_rec, self.niter)

        self.optim_d = Adam(self.d_net.parameters(), self.lr, self.betas)
        self.optim_d_sch = CosineAnnealingLR(self.optim_d, self.niter)


    def save(self):
        torch.save({'net': self.net_gen.state_dict()},
                    f"./{self.save_dir}model_G.torch")
        logging.info(f"Saved to {self.save_dir}model_G.torch")

    def load(self):
        saved = torch.load(f"{self.save_dir}model_G.torch")
        self.net_gen.load_state_dict(saved['net'])
        logging.info(f"Loaded from {self.save_dir}model_G.torch")

    # def _grow_network(self):
    #     self.net_list.append(deepcopy(self.net_list[-1]))
    #     self.net_list[-2].requires_grad_(False)
    #     self.optim = Adam(self.net_list[-1].parameters(), self.lr, self.betas)
    #     self.optim_sch = CosineAnnealingLR(self.opt, self.iter_per_scale)

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

    def _step(self, loss, optim, optim_sch, retain_graph=False):
        optim.zero_grad()
        loss.backward(retain_graph=retain_graph)
        optim.step()
        optim_sch.step()

    def _forward_d(self, img):
        x = img.to(self.opt.device)
        out_x = self.d_net(x)
        return out_x

    def _forward(self, img, code, net):
        x = img.to(self.opt.device)
        l = torch.eye(self.num_classes)[code].to(self.opt.device)
        y = net(x, l)

        return y

    def train_d(self):
        self.d_net.train()
    
        pbar = tqdm(range(self.d_niter))
        for i in pbar:
            x, l = next(self.train_loader_iter)
            output = self._forward_d(x.to(self.opt.device))
            label = torch.eye(self.num_classes)[l].to(self.opt.device)
            print(output, label)
            print(output.shape, label.shape)
            loss = self.loss_ce(output, label)
            self._step(loss, optim=self.optim_d, optim_sch=self.optim_d_sch)

            #pbar.set_postfix(loss=[loss.item()])
            pbar.set_postfix_str(f'loss={loss.item():.4f}')
        del pbar

    def train_self(self):
        torch.autograd.set_detect_anomaly(True)

        pbar = tqdm(range(1, self.niter + 1))
        for iter_cnt in pbar:
            img, l = next(self.train_loader_iter)
            if not img.is_cuda:
                img = img.to(self.opt.device)
                l = l.to(self.opt.device)

            shuffle_l = torch.randperm(len(l)).to(self.opt.device)
            augmented_img = self.aug.transform(img)

            out = self._forward(augmented_img, l, self.net_gen)
            out_label = self.d_net(out)

            loss_ce = self.loss_ce(out_label, torch.eye(self.num_classes)[l].to(self.opt.device))
            loss_ssim = self.loss_ssim(out, img)
            loss_mae = self.loss_mae(out, img)
            loss_lpips = self.loss_lpips(out, augmented_img)

            #loss = loss_lpips.mean()*0.1 + loss_mse*10 +loss_ssim*0.01 #+ loss_ce *0.1
            loss = loss_mae * 10 + loss_lpips.mean()*0.1 + loss_ssim*0.1
            # loss = loss_mae*10 + loss_lpips.mean()*0.1  # best perform currently
            # loss = loss_ssim + loss_mae + loss_ce

            self._step(loss, optim=self.optim_g, optim_sch=self.optim_g_sch, retain_graph=False)

            pbar.set_postfix_str(f'TOT:{loss.item():.4f}, LPIPS:{loss_lpips.mean().item():.4f}, SSIM:{loss_ssim.item():.4f}, MSE:{loss_mae.item():.4f}')
        del pbar

    # def train(self):
    #     start_time = time.time()
    #     start_time_inloop = time.time()

    #     pbar = tqdm(range(1, self.niter + 1))
    #     for iter_cnt in pbar:
    #         img, l = next(self.train_loader_iter)
    #         if not img.is_cuda:
    #             img = img.to(self.opt.device)
    #         if not l.is_cuda:
    #             l = l.to(self.opt.device)
                
    #         x_s, l_s = img[:self.opt.batch_size//2], l[:self.opt.batch_size//2]
    #         x_t, l_t = img[self.opt.batch_size//2:], l[self.opt.batch_size//2:]

    #         out_from_xs = self._forward(x_s, l_t, self.net_gen)
    #         out_label = self.d_net(out_from_xs)

    #         rec_out = self._forward(out_from_xs, l_s, self.net_rec)
    #         rec_label = self.d_net(rec_out)

    #         loss_ce = self.loss_ce(out_label, torch.eye(self.num_classes)[l_t].to(self.opt.device))
    #         loss_ce += self.loss_ce(rec_label, torch.eye(self.num_classes)[l_s].to(self.opt.device))

    #         loss_rec = self.loss_mae(rec_out, x_s)
    #         loss_lpips = self.loss_lpips(x_s, out_from_xs)
    #         loss_lpips += self.loss_lpips(out_from_xs, rec_out)

    #         loss_ssim = self.loss_ssim(x_s, out_from_xs)
    #         loss_ssim += self.loss_ssim(out_from_xs, rec_out)

    #         loss_gen = loss_ce + loss_ssim + loss_lpips.mean()
    #         # loss = loss_ce + loss_rec*10 + loss_lpips.mean() + loss_ssim

    #         self._step(loss, optim=self.optim_g, optim_sch=self.optim_g_sch)
    #         # self._step(loss_ce, optim=self.optim_d, optim_sch=self.optim_d_sch)
            
    #         pbar.set_postfix_str(f'total_loss:{loss.item():.4f}, ce_loss:{loss_ce.item():.4f}, \
    #             rec_loss:{loss_rec.item():.4f}, lpips_loss:{loss_lpips.mean().item():.4f}')
    #     del pbar

    def infer(self):
        pass