import math as m

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Normalize, Resize, RandomCrop
from skimage import color, morphology, filters
from PIL import Image

from utils.functions import interp, convert_image_np
from utils.functions import denorm, norm
from utils.functions import move_to_gpu, move_to_cpu


class ImageLoader:
    def __init__(self, opt):
        self.resampler = interp
        self.init_shape = opt.img_shape
        self.img_path = opt.img_path
        self.opt = opt
        
        self.img = self.load(self.init_shape)
        self.scale_factor, self.scale_num, h, w = self.calc_scale_params(self.img)
        self.img = self.resampler(self.img, (h, w))

        self.img = move_to_gpu(self.img)
        self.img = convert_image_np(self.img)

        #self.imgs = self._multiscale(self.img)
        
        if opt.mask_path is not None:
            self.mask_path = opt.mask_path
            self.mask = self._gen_mask(self.init_shape)
            self.mask = convert_image_np(self.mask)
            #self.masks = self._multiscale(self.mask)
        else:
            self.mask = None
            #self.masks = None
    
    def load(self, init_shape):
        img = Image.open(self.img_path).convert('RGB')
        img = ToTensor()(img)
        img = norm(img)

        if init_shape is not None:
            return self.resampler(img.unsqueeze(0), init_shape)
        else:
            return img.unsqueeze(0)

        return img

    def load_mask(self, init_shape):
        mask = Image.open(self.mask_path).convert('RGB')
        mask = ToTensor()(mask)
        mask = norm(mask)

        if init_shape is not None:
            return self.resampler(mask.unsqueeze(0), init_shape)
        else:
            return mask.unsqueeze(0)

        return mask

    def _multiscale(self, img):
        sf = 1
        imgs = [img]
        *_, h, w = img.shape
        for scale in range(self.scale_num - 1):
            sf = sf * self.scale_factor
            img = self.resampler(img, (round(h * sf), round(w * sf)))
            imgs.append(img)

        return imgs[::-1]

    # Acknowledgement : This code block is a refactored version of that from official SinGAN Repo
    def _gen_mask(self, img_shape):
        def _dilate(mask):
            # adjust radius value
            element = morphology.disk(radius=2)

            mask = mask.squeeze(0).permute(1, 2, 0)
            mask = (mask / 2).clamp(-1, 1) * 255
            mask = mask.cpu().numpy()

            mask = mask.astype(np.uint8)
            mask = mask[:, :, 0]

            mask = morphology.binary_dilation(mask, selem=element)
            mask = filters.gaussian(mask, sigma=5)

            if mask.shape[-1] == 3:
                mask = color.rgb2gray(mask)
            mask = mask[:, :, None, None]
            mask = mask.transpose(3, 2, 0, 1)
            mask = torch.from_numpy(mask).type(torch.cuda.FloatTensor)
            mask = ((mask - 0.5) * 2).clamp(-1, 1)

            mask = mask.expand(1, 3, *mask.shape[-2:])
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            return mask

        mask = denorm(self.load_mask(img_shape))
        mask = _dilate(mask)
        return mask


    def calc_scale_params(self, img):
        *_, h, w = img.shape
        long_side, short_side = max(h, w), min(h, w)

        min_factor = self.opt.min_size / short_side
        max_factor = self.opt.max_size / long_side

        long_side_max, short_side_max = round(long_side * max_factor), round(short_side * max_factor)
        long_side_min, short_side_min = round(long_side * min_factor), round(short_side * min_factor)

        scale_num = m.ceil(m.log(long_side_min / long_side_max, self.opt.scale_factor)) + 1
        new_scale_factor = 1 / m.exp(m.log(long_side_max / long_side_min) * (1 / scale_num))

        if h > w:
            h, w = long_side_max, short_side_max
        else:
            w, h = long_side_max, short_side_max

        return new_scale_factor, scale_num + 1, h, w


class ImageLoader_AUG(ImageLoader):
    def __init__(self, opt, geo_aug, colr_aug):
        super().__init__(opt)
        self.g_aug = geo_aug()
        self.c_aug = colr_aug()

        self.augmented_imgs, self.augmented_masks = self._augment()
        self.dataset = PathologyDataset(self.augmented_imgs, self.augmented_masks,
                        transforms=self.c_aug)
        self.train_dataloader = self._dataloader(self.dataset, self.opt.batch_size)
    
    def _augment(self):
        augmented_imgs = torch.tensor([])
        augmented_masks = torch.tensor([])

        data = {"image": self.img*255.0, "mask": self.mask*255}

        for idx in range(self.opt.num_data):
            geo_augmented = self.g_aug.transform(**data)
            augmented_imgs = torch.cat((augmented_imgs, geo_augmented["image"].unsqueeze(0)), dim=0)

            if self.opt.mask_path is not None:
                #augmented_masks.append(geo_augmented["mask"])
                augmented_masks = torch.cat((augmented_masks, geo_augmented["mask"].unsqueeze(0)), dim=0)

        return augmented_imgs, augmented_masks

    def _dataloader(self, data, batch_size):
        data_loader = iter(
            DataLoader(
                data,
                batch_size = batch_size,
                num_workers = 1,
                sampler = InfiniteSampler(data)
            )
        )
        return data_loader


class PathologyDataset(Dataset):
    def __init__(self, images, masks, transforms=None):
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        
        #_data = {"image": image}
        transformed_image = self.transforms.transform(image=convert_image_np(image))["image"]

        return image, transformed_image, mask
    
    def __len__(self):
        return len(self.images)


class InfiniteSampler(Sampler):
    def __init__(self, data_source):
        super(InfiniteSampler, self).__init__(data_source)
        self.N = len(data_source)

    def __iter__(self):
        while True:
            for idx in torch.randperm(self.N):
                yield idx


class PathologyLoader:
    def __init__(self, opt):
        self.opt = opt
        self.dir_path = opt.dir_path
        self.transform = self._transform()

        self.dataset = self.load_data()
        self.len_ds = self.dataset.__len__()
        self.dataloader = self.make_loader(self.dataset)
        self.dataloader_iter = self.make_iter(self.dataset)


    def load_data(self):
        return ImageFolder(self.dir_path, transform=self.transform)

    def make_loader(self, dataset):
        return DataLoader(dataset, self.opt.batch_size)

    def make_iter(self, dataset):
        data_loader_iter = iter(
            DataLoader(
                dataset,
                batch_size = self.opt.batch_size,
                num_workers = 1,
                sampler = InfiniteSampler(dataset)
            )
        )
        return data_loader_iter

    def _transform(self):
        return transforms.Compose([
                ToTensor(),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Resize(size=self.opt.img_shape)
            ])

