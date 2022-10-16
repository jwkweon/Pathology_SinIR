import torch
import torch.nn.functional as F
import torchvision.transforms as T

import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import VerticalFlip, HorizontalFlip, Flip, RandomCrop, Rotate, Resize
from albumentations import HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, OneOf,\
    Compose, MultiplicativeNoise, ToSepia, ChannelDropout, ChannelShuffle, Cutout, InvertImg

class GeoAugment():
    ''' Augmentation for input image & mask

        Augmentation list:
            [Flip, Rotate, RandomCrop, Resize]

        Usage:
            g_aug = GeoAugment()
            INPUT : 
                data = {"image": x, "mask": m}  # x,m = [0, 255]
            augmented = g_aug.transform(**data)
            OUTPUT : 
                aug_image = augmented["image"]  # [-1, 1]
                aug_mask = augmented["mask"]
    '''
    def __init__(self):
        super().__init__()
        self._transform = self.geometric_aug()
        
    def geometric_aug(self):
        return Compose([
        Flip(),
        Rotate(),
        RandomCrop(256, 256),
        Resize(256, 256),
        A.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
        ToTensorV2(),
        ],
        additional_targets={'image': 'image', 'mask': 'image'},
        )
    
    def transform(self, **x):
        #_transform = self.geometric_aug()
        return self._transform(**x)


class Augment():
    ''' Augmentation for input image

        Augmentation list:
            [[MultiplicativeNoise, IAAAdditiveGaussianNoise, GaussNoise]
                [InvertImg, ToSepia]        # del InvertImg because of input value range 
                [ChannelDropout, ChannelShuffle]]
            [Cutout]

        Usage:
            aug = Augment()
            INPUT : 
                data = {"image": x}  # x = [-1, 1]
            augmented = aug.transform(**data)
            OUTPUT : 
                aug_image = augmented["image"] # [-1, 1]
    '''
    def __init__(self):
        super().__init__()
        self._transform = self.strong_aug()
        
    def strong_aug(self):
        color_r = random.uniform(0, 1)
        color_g = random.uniform(0, 1)
        color_b = random.uniform(0, 1)
        
        return Compose([
            OneOf([
                OneOf([
                    MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, per_channel=True, p=0.2),
                    GaussNoise()]),
                OneOf([
                    ChannelDropout(channel_drop_range=(1, 1), fill_value=0),
                    ChannelShuffle()]),
                #ToSepia(),
                HueSaturationValue(hue_shift_limit=25, sat_shift_limit=0.2, val_shift_limit=0, p=0.1)],
                p=0.25),
            #Cutout(num_holes=2, max_h_size=30, max_w_size=30,
            #       fill_value=[color_r, color_g, color_b], p=0.9),
            #A.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
            ToTensorV2(),
            ],
            #additional_targets={'image0': 'image', 'image1': 'image', 'image2': 'image', 'image3': 'image'}
        )
    
    def transform(self, **x):
        return self._transform(**x)

class JitAugment():
    # Modify Differentiable Augmentation for Data-Efficient GAN Training
    # Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
    # https://arxiv.org/pdf/2006.10738
    def __init__(self, policy=''):
        super().__init__()
        self.augment_fns = {
            'color': [self.rand_brightness, self.rand_saturation, self.rand_contrast],
            'translation': [self.rand_translation],
            'cutout': [self.rand_cutout],
            'shuffle': [self.shuffle_pixel],
        }
        self.policy = policy
        #self._transfrom = self.jit_aug()


    def rand_brightness(self, x):
        x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
        return x

    def rand_saturation(self, x):
        x_mean = x.mean(dim=1, keepdim=True)
        x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
        return x

    def rand_contrast(self, x):
        x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
        return x

    def rand_translation(self, x, ratio=0.125):
        shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
        translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(2), dtype=torch.long, device=x.device),
            torch.arange(x.size(3), dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
        grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
        x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
        x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
        return x

    def rand_cutout(self, x, ratio=0.5):
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
        offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
        return x

    def shuffle_pixel(self, x, p=0.005):
        if p == 0:
            return x.clone()

        b, c, h, w = x.shape
        out = x.clone()
        original_idx = torch.arange(h * w)
        shuffle_idx = torch.randperm(h * w)
        shuffle_idx = torch.where(torch.rand(*original_idx.shape) > p, original_idx, shuffle_idx)

        out = out.view(b, 3, -1)[:, :, shuffle_idx].view(*out.shape)
        return out

    def transform(self, x):
        if self.policy:
            for p in self.policy.split(','):
                for f in self.augment_fns[p]:
                    x = f(x)
            x = x.contiguous()
        return x

    # def transform(self, **x):
    #     return self._transfrom(**x)

class RandAug():
    def __init__(self):
        super().__init__()
        self.rand_hflip = T.RandomHorizontalFlip(0.2)
        self.rand_vflip = T.RandomVerticalFlip(0.2)
        self.rand_rot = T.RandomRotation(15)
        self.rand_crp_rsz = T.RandomResizedCrop((256, 256), (0.5,1))

    def transform(self, x):
        x = self.rand_hflip(x)
        x = self.rand_vflip(x)
        x = self.rand_rot(x)
        x = self.rand_crp_rsz(x)

        return x
