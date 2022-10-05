import torch
import torch.nn as nn
import torch.nn.modules.conv as conv
from torchvision.models import squeezenet1_1
import torch.nn.functional as F

from model.blocks import *


class Network(nn.Module):
    def __init__(self, img_ch, net_ch):
        super().__init__()

        self.from_rgb = nn.Sequential(
            nn.Conv2d(img_ch, net_ch // 2, 1, 1, 0),
            nn.Conv2d(net_ch // 2, net_ch, 1, 1, 0)
        )
        self.to_rgb = nn.Sequential(
            nn.Conv2d(net_ch, net_ch // 2, 1, 1, 0),
            nn.Conv2d(net_ch // 2, img_ch, 1, 1, 0),
            nn.Tanh()
        )
        self.conv_block = ConvBlock(net_ch, net_ch, norm='in', act='leakyrelu')

        self.layers = nn.Sequential(
            *[self.conv_block for _ in range(6)]
        )

    def forward(self, x):
        x = self.from_rgb(x)

        dense = [x]
        for l in self.layers:
            x = l(x)
            for d in dense:
                x = x + d

        x = self.to_rgb(x)
        return x


class BGNetwork(nn.Module):
    ''' z_dim : 100
        img_ch : 3
        net_ch : 64  # opt.nfg
    '''
    def __init__(self, z_dim, img_ch, net_ch):
        super().__init__()
        self.init_ch = net_ch * 32

        self.fc = nn.Sequential(
            nn.Linear(z_dim, self.init_ch * 8 * 8 * 2, bias=False),
            nn.BatchNorm1d(self.init_ch * 8 * 8 * 2),
            GLU()
        )

        self.to_rgb = nn.Sequential(
            nn.Conv2d(net_ch, net_ch // 2, 1, 1, 0),
            nn.Conv2d(net_ch // 2, img_ch, 1, 1, 0),
            nn.Tanh()
        )

        self.layers = nn.Sequential(
            *[upBlock(self.init_ch//(2**i), self.init_ch//(2**(i+1))) for i in range(5)]
        )
    
    def forward(self, z_code):
        in_code = self.fc(z_code)
        out_code = in_code.view(-1, self.init_ch, 8, 8)

        for l in self.layers:
            out_code = l(out_code)
        # out_code = self.upsample1(out_code)
        # out_code = self.upsample2(out_code)
        # out_code = self.upsample3(out_code)
        # out_code = self.upsample4(out_code)
        # out_code = self.upsample5(out_code)
        # out_code = self.upsample6(out_code)

        out_bg_img = self.to_rgb(out_code)

        return out_bg_img


class NetworkWithCode(nn.Module):
    def __init__(self, img_ch, net_ch, n_class=4):
        super(NetworkWithCode, self).__init__()
        self.n_class = n_class

        self.from_rgb = nn.Sequential(
            nn.Conv2d(img_ch+1, net_ch // 2, 1, 1, 0),
            nn.Conv2d(net_ch // 2, net_ch, 1, 1, 0)
        )
        self.to_rgb = nn.Sequential(
            nn.Conv2d(net_ch, net_ch // 2, 1, 1, 0),
            nn.Conv2d(net_ch // 2, img_ch, 1, 1, 0),
            nn.Tanh()
        )
        self.from_code = nn.Sequential(
            # nn.Embedding(self.n_class, 4),
            nn.Linear(self.n_class, 128),
            nn.Linear(128, 1024),
            nn.Linear(1024, 256*256)
        )

        self.conv_block = ConvBlock(net_ch, net_ch, norm='in', act='leakyrelu')

        self.layers = nn.Sequential(
            *[self.conv_block for _ in range(3)]
        )
    
    def forward(self, x, code):
        e_code = self.from_code(code)
        code_img = e_code.view(-1, 1, 256, 256)
        x = torch.cat((x, code_img), dim=1)
        x = self.from_rgb(x)
        
        dense = [x]
        for l in self.layers:
            x = l(x)
            for d in dense:
                x = x + d
        x = self.to_rgb(x)

        return x



class SqueezeNet(nn.Module):
    def __init__(self, n_class=4):
        super(SqueezeNet, self).__init__()
        self.n_class = n_class
        self.base_model = squeezenet1_1(pretrained=True)
        temp = squeezenet1_1(pretrained=False, num_classes=n_class)
        self.base_model.classifier = temp.classifier
        del temp

    def forward(self, x):
        x = self.base_model(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, img_ch, net_ch):
        super().__init__()

        self.from_rgb = nn.Sequential(
            nn.Conv2d(img_ch, net_ch // 2, 1, 1, 0),
            nn.Conv2d(net_ch // 2, net_ch, 1, 1, 0)
        )

    
    def forward(self, x):
        out = self.from_rgb(x)

        return out