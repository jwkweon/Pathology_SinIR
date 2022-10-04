import torch
import torch.nn as nn
import torch.nn.functional as F

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, ker_s=3, stride=1, padding=0, norm='in',
                act='leakyrelu'):
        super(ConvBlock, self).__init__()

        if norm == 'in':
            self.norm = nn.InstanceNorm2d(out_c)
        elif norm == 'bn':
            self.norm = nn.BatchNorm2d(out_c)
        elif norm == 'identity':
            self.norm = nn.Identity()

        if act == 'relu':
            self.act = nn.ReLU(True)
        elif act == 'leakyrelu':
            self.act = nn.LeakyReLU(inplace = True, negative_slope = 0.2)
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'identity':
            self.act = nn.Identity()

        self.layer = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_c, out_c, ker_s, stride, padding),
            self.norm,
            self.act
        )

    def forward(self, x):
        return self.layer(x)

def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False),
        nn.BatchNorm2d(out_planes),
        #GLU()
        nn.LeakyReLU(inplace = True, negative_slope = 0.2)
    )
    return block