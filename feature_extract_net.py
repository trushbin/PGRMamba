from torch import nn
import torch
from collections import OrderedDict
import math
from einops import rearrange
import torch.nn.functional as F
import os
import sys
import numpy as np

from mamba.mamba import G_SAM, Conv3D




class SFE(nn.Module):
    def __init__(self, channels, size=5, sigma1=0.8, sigma2=1.6):
        super().__init__()
        self.channels = channels
        self.size = size
        self.dog_conv = nn.Conv3d(channels, channels, kernel_size=size, 
                                  padding=size//2, groups=channels, bias=False)
        self._init_dog_weights(sigma1, sigma2)
        self.proj = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.BatchNorm3d(channels),
            nn.ReLU()
        )

    def _init_dog_weights(self, s1, s2):
        size = self.size
        ax = torch.linspace(-(size-1)/2., (size-1)/2., size)
        zz, yy, xx = torch.meshgrid(ax, ax, ax, indexing='ij')
        dist_sq = xx**2 + yy**2 + zz**2
        
        # 生成两个高斯核之差
        g1 = torch.exp(-0.5 * dist_sq / s1**2)
        g1 /= (g1.sum() + 1e-8)
        
        g2 = torch.exp(-0.5 * dist_sq / s2**2)
        g2 /= (g2.sum() + 1e-8)
        
        dog_kernel = (g1 - g2).view(1, 1, size, size, size)
        
        with torch.no_grad():
            self.dog_conv.weight.copy_(dog_kernel.repeat(self.channels, 1, 1, 1, 1))
            self.dog_conv.weight.requires_grad = True

    def forward(self, x):
        feat = self.dog_conv(x)
        out = self.proj(feat)
        return out







class DSFB_1(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        self.filters2 = SFE(inplanes, size=9)
        
        self.conv1 = nn.Conv3d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv3d(planes[0], planes[1], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        x2 = self.filters2(x)
        out = x2 + x

        residual = out
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out += residual
        return out




class DSFB_2(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        self.filters2 = SFE(inplanes, size=5)

        self.conv1 = nn.Conv3d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv3d(planes[0], planes[1], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        x2 = self.filters2(x)
        out = x2 + x

        residual = out
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out += residual
        return out


class Focus(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.focus1 = nn.AvgPool3d(kernel_size=4, stride=4)  
        self.focus2 = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x1 = self.focus1(x1)
        x2 = self.focus2(x2)
        y = torch.cat((x1, x2), 1)
        z = torch.norm(y, dim=1, keepdim=True) 
        return z


class Model_net(nn.Module):
    def __init__(self, layers):
        super(Model_net, self).__init__()
        self.inplanes = 16
        self.conv_net1 = nn.Sequential(
            nn.Conv3d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(self.inplanes),
            nn.LeakyReLU(0.1)
        )
        self.layer1 = self._make_layer1([16, 32], layers[0])
        self.layer2 = self._make_layer2([32, 64], layers[1])
        self.conv_net2 = nn.Sequential(
            nn.Conv3d(64, 8, kernel_size=1),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(0.1)
        )
        self.mamba = G_SAM(8, 8)
        self.conv_net3 = nn.Conv3d(8, 4,kernel_size=1)
        
        self.focus = Focus(16, 32)


    def _make_layer1(self, planes, blocks):
        layers = []
        layers.append(("ds_conv", nn.Conv3d(self.inplanes, planes[1], kernel_size=3,
                                stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm3d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))

        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), DSFB_1(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))

    def _make_layer2(self, planes, blocks):
        layers = []
        layers.append(("ds_conv", nn.Conv3d(self.inplanes, planes[1], kernel_size=3,
                                stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm3d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))

        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), DSFB_2(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))


    def forward(self, x):
        x1 = self.conv_net1(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.conv_net2(x3)
        
        y = self.focus(x1, x2)      
          
        x5 = self.mamba(x4, y) 
        out = self.conv_net3(x5)
        return out





if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = torch.rand([1, 1, 256, 256, 256]).float().to(device)
    net = Model_net(layers=[3, 3]).to(device)
    print(net(image).shape)


