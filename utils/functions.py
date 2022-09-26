import os
import time
import random
import numpy as np
import datetime
import dateutil.tz

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def norm(x):
    out = (x - 0.5) * 2
    return out.clamp(-1, 1)

def move_to_gpu(t):
    if (torch.cuda.is_available()):
        t = t.to(torch.device('cuda'))
    return t

def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t

def convert_image_np(inp):
    if len(inp.shape) == 4:
        if inp.shape[1]==3:
            inp = denorm(inp)
            inp = move_to_cpu(inp[-1,:,:,:])
            inp = inp.numpy().transpose((1,2,0))
        else:
            inp = denorm(inp)
            inp = move_to_cpu(inp[-1,-1,:,:])
            inp = inp.numpy().transpose((0,1))
    else:
        if inp.shape[0]==3:
            inp = denorm(inp)
            inp = move_to_cpu(inp[:,:,:])
            inp = inp.numpy().transpose((1,2,0))
        else:
            inp = denorm(inp)
            inp = move_to_cpu(inp[-1,:,:])
            inp = inp.numpy().transpose((0,1))

    inp = np.clip(inp,0,1)
    return inp

def np2torch(x, opt):
    if opt.nc_im == 3:
        x = x[:,:,:,None]
        x = x.transpose((3, 2, 0, 1))/255
    else:
        x = color.gray2rgb(x)   # to Harmonization
        x = x[:,:,:,None]
        x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x)
    if not(opt.not_cuda):
        x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if not(opt.not_cuda) else x.type(torch.FloatTensor)
    x = norm(x)
    return x

def generate_dir2save(opt):
    training_image_name = opt.img_path[:-4].split("/")[-1]
    dir2save = 'TrainedModels/{}/'.format(training_image_name)
    dir2save += opt.timestamp
    #dir2save += "_{}".format(opt.train_mode)

    return dir2save


def interp(x, img_shape):
    if isinstance(img_shape, (tuple, list)):
        return F.interpolate(x, size=img_shape, mode='bicubic', align_corners=True).clamp(-1, 1)
    elif isinstance(img_shape, int):
        h, w = x.shape[-2:]
        sf = img_shape / max(h, w)
        h, w = int(h * sf), int(w * sf)
        return F.interpolate(x, size=(h, w), mode='bicubic', align_corners=True).clamp(-1, 1)
    else:
        raise Exception


def post_config(opt):
    # init fixed parameters
    opt.device = torch.device("cpu" if opt.not_cuda else "cuda:{}".format(opt.gpu))
    opt.timestamp = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt
