import torch
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
