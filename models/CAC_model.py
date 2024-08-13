import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet_parts import *
from models.cbam import *
# from unet_parts import *
# from cbam import *
import copy
import math

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class Generator(nn.Module):
    def __init__(self, img_size=64, style_dim=5, in_ch=2,out_ch=1):
        super().__init__()
        dim_in = 2**12 // img_size
        gate_channels = dim_in
        self.img_size = img_size
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(gate_channels//2, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(gate_channels//2, out_ch, 1, 1, 0))
        bilinear=False
        self.inc = DoubleConv(in_ch, gate_channels//2)
        self.down1 = Down(gate_channels//2, gate_channels)
        self.down2 = Down(gate_channels, gate_channels*2)
        self.attn1 = CBAM(gate_channels=gate_channels//2)
        self.attn2 = CBAM(gate_channels=gate_channels)
        self.dropout = nn.Dropout2d(0.5)
        self.style1 = AdaIN(style_dim, gate_channels*2)
        self.style2 = AdaIN(style_dim, gate_channels)
        self.style3 = AdaIN(style_dim, gate_channels//2)
        self.up2 = Up(gate_channels*2, gate_channels, bilinear,dp=True)
        self.up1 = Up(gate_channels, gate_channels//2, bilinear,dp=True)
        self.outc = DoubleConv(gate_channels//2, gate_channels//2)

    def forward(self, x, s, masks=None):
        x1 = self.inc(x)
        x1a = self.attn1(x1)
        # x1a = self.dropout(x1a)
        x2 = self.down1(x1)
        x2a = self.attn2(x2)
        # x2a = self.dropout(x2a)
        x3 = self.down2(x2)
        # x3 = self.dropout(x3)

        # add style code
        x3 = self.style1(x3,s)
        x  = self.up2(x3,x2a)
        x = self.style2(x,s)
        x = self.up1(x,x1a)
        x = self.style3(x,s)
        x = self.outc(x)

        return self.to_rgb(x)


class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=1, max_conv_dim=256, in_ch=2):
        super().__init__()
        dim_in = 2**12 // img_size
        blocks = []
        blocks += [nn.Conv2d(in_ch, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.main(x)
        # out = out.view(out.size(0), -1)  # (batch, num_domains)
        # idx = torch.LongTensor(range(y.size(0))).to(y.device)
        # out = out[idx, y]  # (batch)
        return torch.sigmoid(out)

if __name__=="__main__":
	x = torch.rand(5, 1, 64, 64)
	netD = Discriminator(img_size=64, in_ch=1)	
	out_discriminator = netD(x)

	print("Discriminator shape: ",out_discriminator.shape)

	y = torch.rand(5, 2, 64, 64)
	s = torch.rand(5, 5)
	netG = Generator(in_ch=2,out_ch=1)
	out_gen = netG(y,s)
	print("Generator output shape: ",out_gen.shape)


