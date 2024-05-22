import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cams import CAM

"""
Architecture based on InfoGAN paper.
"""

class Generator(nn.Module):
	def __init__(self,in_channels=12):
		super().__init__()

		self.cam = CAM(in_channels,64)
	def forward(self, x):
		img = F.relu(self.cam(x))

		return img

class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv2d(1, 64, 4, 2, 1)

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
	x = torch.rand(23, 1, 72, 72)
	model = Discriminator()	
	netD  = DHead()
	out_discriminator = model(x)
	probs = netD(out_discriminator)

	
	print("netD Shape: ",probs.shape)
	print("Discriminator shape: ",out_discriminator.shape)

	y = torch.rand(1, 12, 72, 72)
	netG = Generator()
	netQ = QHead()
	out_gen = netG(y)
	print("Generator output shape: ",out_gen.shape)

	q_logits, q_mu, q_var = netQ(out_discriminator)

	print("netQ shape: ",q_logits.shape)

