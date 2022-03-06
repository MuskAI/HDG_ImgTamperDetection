""" Full assembly of the parts to form the complete network """
import torch
import torch.nn.functional as F
from torchsummary import summary
import sys
sys.path.append('.')
import torch.nn as nn
from unet_parts import *
from aspp import ASPP
from srm import SRMConv


class UNet(nn.Module):
    def __init__(self, n_channels=3, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = 1
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels+3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 1)


        ## adding aspp
        self.aspp = ASPP(1024 // factor,[6,12,18,24],1024 // factor)

        ## adding srm pre conv
        self.srm = SRMConv(cuda=False)


    def forward(self, x):

        # pre srm conv
        srm = self.srm(x)
        # print(srm.size())
        # concat srm and rgb
        # x0 is 6 c
        x0 = torch.cat([srm,x], dim=1)
        # print(x0.size())
        x1 = self.inc(x0)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        aspp = self.aspp(x5)
        x5 = aspp
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    model = UNet(3,bilinear=True).cpu()
    summary(model,(3,320,320),device='cpu',batch_size=5)