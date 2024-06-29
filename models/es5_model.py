import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet_parts import *
from models.cbam import *
# from unet_parts import *
# from cbam import *

"""
Architecture based on InfoGAN paper.
"""

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
	def __init__(self,in_channels=1):
		super().__init__()

		self.conv1 = nn.Conv2d(in_channels, 64, 4, 2, 1)

		self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
		self.bn2 = nn.BatchNorm2d(128)

		self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
		self.bn3 = nn.BatchNorm2d(256)

	def forward(self, x):
		x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
		x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
		x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)

		return x

class DHead(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv = nn.Conv2d(256, 1, 8)

	def forward(self, x):
		output = torch.sigmoid(self.conv(x))

		return output

class QHead(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv2d(256, 128, 8, bias=False)
		self.bn1 = nn.BatchNorm2d(128)

		self.conv_disc = nn.Conv2d(128, 11, 1)

		self.conv_mu = nn.Conv2d(128, 1, 1)
		self.conv_var = nn.Conv2d(128, 1, 1)

	def forward(self, x):
		x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)

		disc_logits = self.conv_disc(x).squeeze()

		# Not used during training for celeba dataset.
		mu = self.conv_mu(x).squeeze()
		var = torch.exp(self.conv_var(x).squeeze())

		return disc_logits, mu, var

if __name__=="__main__":
	x = torch.rand(5, 2, 64, 64)
	model = Discriminator(2)	
	netD  = DHead()
	out_discriminator = model(x)
	probs = netD(out_discriminator)

	
	print("netD Shape: ",probs.shape)
	print("Discriminator shape: ",out_discriminator.shape)

	y = torch.rand(5, 2, 64, 64)
	s = torch.rand(5, 5)
	netG = Generator(in_ch=2,out_ch=1)
	netQ = QHead()
	out_gen = netG(y,s)
	print("Generator output shape: ",out_gen.shape)

	q_logits, q_mu, q_var = netQ(out_discriminator)

	print("netQ shape: ",q_logits.shape)

