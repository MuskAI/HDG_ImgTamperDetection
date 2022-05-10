""" Full assembly of the parts to form the complete network """
import torch
import torch.nn.functional as F
from torchsummary import summary
import sys
sys.path.append('.')
import torch.nn as nn
from .unet_parts import *
from .aspp import ASPP
from .srm import SRMConv
from .grid_attention_layer import GridAttentionBlock2D as attention_gate


class UNet(nn.Module):
    def __init__(self, n_channels=3, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = 1
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.scale_factor = 1
        self.channel_list = [32 // self.scale_factor, 64 // self.scale_factor, 128 // self.scale_factor,
                             256 // self.scale_factor, 512 // self.scale_factor]

        self.cls_extra_channel_list = [128 // self.scale_factor, 32 // self.scale_factor]

        self.inc = DoubleConv(n_channels+3, self.channel_list[0])
        self.down1 = Down(self.channel_list[0], self.channel_list[1])
        self.down2 = Down(self.channel_list[1], self.channel_list[2])
        self.down3 = Down(self.channel_list[2], self.channel_list[3])
        self.down4 = Down(self.channel_list[3], self.channel_list[4] // factor)

        self.up1 = Up(self.channel_list[4], self.channel_list[3] // factor, bilinear)
        self.up2 = Up(self.channel_list[3], self.channel_list[2] // factor, bilinear)
        self.up3 = Up(self.channel_list[2], self.channel_list[1] // factor, bilinear)
        self.up4 = Up(self.channel_list[1], self.channel_list[0], bilinear)
        self.up4_band = Up(self.channel_list[1], self.channel_list[0], bilinear)
        self.outc = OutConv(self.channel_list[0], 1)
        self.outc_band = OutConv(self.channel_list[0], 1)

        ## adding aspp
        self.aspp = ASPP(self.channel_list[4] // factor,[1,2,3,4],self.channel_list[4] // factor)

        ## adding srm pre conv
        self.srm = SRMConv(cuda=False)

        ## adding attention gate module ,top-down order
        self.attention_gate1 = attention_gate(in_channels=self.channel_list[0],gating_channels=self.channel_list[1]//factor,sub_sample_factor=(2,2))
        self.attention_gate2 = attention_gate(in_channels=self.channel_list[1],gating_channels=self.channel_list[2]//factor,sub_sample_factor=(2,2))
        self.attention_gate3 = attention_gate(in_channels=self.channel_list[2],gating_channels=self.channel_list[3]//factor,sub_sample_factor=(2,2))
        self.attention_gate4 = attention_gate(in_channels=self.channel_list[3],gating_channels=self.channel_list[4] // factor,sub_sample_factor=(2,2))

        ## adding semi-supervised branch
        self.clsnet = ClsNet(in_channels=self.channel_list[4] // factor,  n_classes=1,scale_factor=2)

        ## decrease dims
    def forward(self, x):

        # pre srm conv
        srm = self.srm(x)
        # print(srm.size())
        # concat srm and rgb
        # x0 is 6 c
        x0 = torch.cat([srm,x], dim=1)
        # print(x0.size())
        x1 = self.inc(x0)

        x2 = self.down1(x1) # 1/2
        x3 = self.down2(x2) # 1/4
        x4 = self.down3(x3) # 1/8
        x5 = self.down4(x4) # 1/16
        aspp = self.aspp(x5)
        x5 = aspp
        x_cls = self.clsnet(x5)
        # print('### gate4:',x4.size(),x5.size())
        x4 = self.attention_gate4(x4, x5)[0]
        x = self.up1(x5, x4)

        x3 = self.attention_gate3(x3, x)[0]
        x = self.up2(x, x3)

        x2 = self.attention_gate2(x2, x)[0]
        x = self.up3(x, x2)

        x1 = self.attention_gate1(x1, x)[0]
        x_area = self.up4(x, x1)

        x_band = self.up4_band(x, x1)
        x_band = self.outc_band(x_band)
        x_area = self.outc(x_area)
        return x_area,x_band,x_cls


class ClsNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=1,scale_factor=2):
        """
        implement semi cls supervised branch
        this model is same with the unet encoder part,but smaller than the unet seg part
        :param n_channels:
        :param bilinear:
        :param n_classes:
        """
        super(ClsNet, self).__init__()

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.scale_factor = scale_factor
        self.cls_extra_channel_list = [128//self.scale_factor,32//self.scale_factor]


        # the input scale is 20 *20

        self.extra1 = nn.Conv2d(in_channels=self.in_channels,
                      out_channels=self.cls_extra_channel_list[0], kernel_size=3, stride=2, padding=1)
        # 10*10
        self.extra2 = nn.Conv2d(in_channels=self.cls_extra_channel_list[0],
                      out_channels=self.cls_extra_channel_list[1], kernel_size=3, stride=2, padding=1)
        # 5*5
        self.extra3 = nn.AdaptiveAvgPool2d(1)

        # 1*1
        self.extra4 = nn.Conv2d(in_channels=self.cls_extra_channel_list[1],out_channels=1,kernel_size=1)


    def forward(self, x):

        """
        :param x: rgb with 3 channel
        :return:
        """
        x = self.extra1(x)
        x = self.extra2(x)
        x = self.extra3(x)
        x = self.extra4(x)
        x = x.view((x.shape[0],1))
        # x = nn.Linear(x.shape[1],out_features=1)(x)
        x = torch.nn.Sigmoid()(x)
        return x


if __name__ == '__main__':
    model = UNet(3,bilinear=True).cpu()
    summary(model,(3,320,320),device='cpu',batch_size=2)