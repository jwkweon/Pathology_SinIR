import os
import sys
import argparse
# import shutil
# import logging
from glob import glob

import torch
from torchvision.utils import save_image

from utils.loader import PathologyLoader 
from utils.functions import post_config, generate_dir2save
from utils.runner import *
from config.config import get_arguments

if __name__ == '__main__':
    parser = get_arguments()
    #parser.add_argument('--img_path', help='input slide image for training', required=True)
    parser.add_argument('--dir_path', help='input image dir for training', required=True)
    parser.add_argument('--dir_test_path', help='test image dir', default=None)
    parser.add_argument('--gpu', type=int, help='which GPU to use', default=0)
    parser.add_argument('--img_shape', help='target size of input image', default=(256, 256))
    # Inference
    #parser.add_argument('--mask_path', help='mask dir for training', required=True)

    opt = parser.parse_args()
    opt = post_config(opt)

    if not os.path.exists(opt.dir_path):
        print("Image does not exist: {}".format(opt.dir_path))
        print("Please specify a valid image.")
        exit()

    if torch.cuda.is_available():
        #device = torch.device("cuda")
        torch.cuda.set_device(opt.gpu)

    dir2save = generate_dir2save(opt)
    
    if os.path.exists(dir2save):
        print('Trained model already exist: {}'.format(dir2save))
        exit()

    # create log dir
    try:
        os.makedirs(dir2save)
    except OSError:
        pass

    # train model
    runner = Runner(PathologyLoader, dir2save, opt)
    runner.train_d()
    runner.train()
    # python main.py --img_path 'imgs/path.png' --mask_path 'imgs/mask_path.png' --gpu 0
    