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
        self.channel_list = [32,64,128,256,512]

        self.inc = DoubleConv(n_channels+3, self.channel_list[0])
        self.down1 = Down(self.channel_list[0], self.channel_list[1])
        self.down2 = Down(self.channel_list[1], self.channel_list[2])
        self.down3 = Down(self.channel_list[2], self.channel_list[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(self.channel_list[3], self.channel_list[4] // factor)
        self.up1 = Up(self.channel_list[4], self.channel_list[3] // factor, bilinear)
        self.up2 = Up(self.channel_list[3], self.channel_list[2] // factor, bilinear)
        self.up3 = Up(self.channel_list[2], self.channel_list[1] // factor, bilinear)
        self.up4 = Up(self.channel_list[1], self.channel_list[0], bilinear)
        self.outc = OutConv(self.channel_list[0], 1)


        ## adding aspp
        self.aspp = ASPP(self.channel_list[4] // factor,[6,12,18,24],self.channel_list[4] // factor)

        ## adding srm pre conv
        self.srm = SRMConv(cuda=False)

        ## adding attention gate module ,top-down order
        self.attention_gate1 = attention_gate(in_channels=self.channel_list[0],gating_channels=self.channel_list[1]//factor,sub_sample_factor=(2,2))
        self.attention_gate2 = attention_gate(in_channels=self.channel_list[1],gating_channels=self.channel_list[2]//factor,sub_sample_factor=(2,2))
        self.attention_gate3 = attention_gate(in_channels=self.channel_list[2],gating_channels=self.channel_list[3]//factor,sub_sample_factor=(2,2))
        self.attention_gate4 = attention_gate(in_channels=self.channel_list[3],gating_channels=self.channel_list[4] // factor,sub_sample_factor=(2,2))


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
        # print('### gate4:',x4.size(),x5.size())
        x4 = self.attention_gate4(x4, x5)[0]
        x = self.up1(x5, x4)

        x3 = self.attention_gate3(x3, x)[0]
        x = self.up2(x, x3)

        x2 = self.attention_gate2(x2, x)[0]
        x = self.up3(x, x2)

        x1 = self.attention_gate1(x1, x)[0]
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    model = UNet(3,bilinear=True).cpu()
    summary(model,(3,320,320),device='cpu',batch_size=2)