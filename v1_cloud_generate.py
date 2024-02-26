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

noise1 , _ = noise_sample(10,10,0,128,9,device)

# Generate image.
with torch.no_grad():
    generated_img1 = netG(noise1).detach().cpu()
generated_img1 = generated_img1.numpy()
fname = "v1_cloud/"+"gen_img.png"
plot_cot2(generated_img1[0,0],"Radiance at 0.66um",fname,False,[0,2])
