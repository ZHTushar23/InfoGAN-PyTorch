import argparse

import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', required=True, help='Checkpoint to load path from')
args = parser.parse_args()

from models.cloud_model import Generator
from utils import *
from visualization import *

# Load the checkpoint file
state_dict = torch.load(args.load_path)

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Get the 'params' dictionary from the loaded state_dict.
params = state_dict['params']

# Create the generator network.
netG = Generator().to(device)
# Load the trained generator weights.
netG.load_state_dict(state_dict['netG'])
print(netG)

# noise1 , _ = noise_sample(10,10,0,128,9,device)

idx = np.arange(3).repeat(3)
dis_c = torch.zeros(9, 10, 10, device=device)
dis_c[torch.arange(0, 9), idx] = 1.0
# Discrete latent code.
c1 = dis_c.view(9, -1, 1, 1)
z = torch.randn(9, 128, 1, 1, device=device)
noise1 = torch.cat((z, c1), dim=1)



# Generate image.
with torch.no_grad():
    generated_img1 = netG(noise1).detach().cpu()
generated_img1 = generated_img1.numpy()
fname = "v1_cloud/"+"gen_img.png"
plot_cot2(generated_img1[0,0],"Radiance at 0.66um",fname,False,[0,2])

# b = dis_c.detach().cpu().numpy()
# print(b.shape)
# plt.imshow(b[5])
# plt.colorbar()
# plt.savefig('demo.png')
# plt.close()