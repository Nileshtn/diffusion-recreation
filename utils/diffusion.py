import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class Diffusion(nn.Module):
    def __init__(self, in_channel, out_channel, time_steps):
        super(Diffusion, self).__init__()
        self.time_embedding = TimeEmbbeding(time_steps, 8)
        self.d_conv1 = DoubleConv(in_channel, 32)

        self.downs1 = DownSample(32, 64, 8)
        self.d_att1 = SelfAttention(64, 8, 2)

        self.downs2 = DownSample(64, 128, 8)
        self.d_att2 = SelfAttention(128, 8, 2)

        self.downs3 = DownSample(128, 128, 8)
        self.d_att3 = SelfAttention(128, 8, 2)

        self.bn1 = DoubleConv(128, 512)
        self.bn2 = DoubleConv(512, 512)
        self.bn3 = DoubleConv(512, 128)

        self.ups1 = UpSample(256, 64, 8)
        self.u_att1 = SelfAttention(64, 8, 2) 

        self.ups2 = UpSample(128, 32, 8)
        self.u_att2 = SelfAttention(32, 8, 2) 

        self.ups3 = UpSample(64, 32, 8)
        self.u_att3 = SelfAttention(32, 8, 2)

        self.out_conv = nn.Conv2d(32, out_channel, 1)

    def forward(self, x, t):
        tm = self.time_embedding(t)
        x1 = self.d_conv1(x) # -> 32
        x = self.downs1(x1, tm)
        x2 = self.d_att1(x) # -> 64
        x = self.downs2(x2, tm)
        x3 = self.d_att2(x) # -> 128
        x = self.downs3(x, tm)
        x = self.d_att3(x)
        x = self.bn1(x)
        x = self.bn2(x)
        x = self.bn3(x) # -> 128
        x = self.ups1(x, x3, tm) # -> 64
        x = self.u_att1(x)
        x = self.ups2(x, x2, tm)
        x = self.u_att2(x)
        x = self.ups3(x, x1, tm)
        x = self.u_att3(x)
        x = self.out_conv(x)        
        return x
