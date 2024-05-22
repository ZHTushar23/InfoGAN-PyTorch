import argparse

import torch
import torchvision.utils as vutils
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', required=True, help='Checkpoint to load path from')
args = parser.parse_args()

from models.cloud_model18 import Generator
from v2_utils import *
from visualization import *
from v18_dataset import NasaDataset
from v18_config import params

# Load the checkpoint file
state_dict = torch.load(args.load_path)

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Get the 'params' dictionary from the loaded state_dict.
# model_params = state_dict['params']

# Create the generator network.
netG = Generator(13).to(device)
# Load the trained generator weights.
netG.load_state_dict(state_dict['netG'])
# print(netG)

# noise1 , _ = noise_sample(10,10,0,128,9,device)

dataset_dir1 = "/home/local/AD/ztushar1/Data/LES_vers1_multiangle_results"
# sza_list = [60.0,40.0,20.0,4.0]
vza_list1 = params['vza_list1']
vza_list2 = params['vza_list2']
sza_list1 = params['sza_list1']
sza_list2 = params['sza_list2']

f = np.arange(1,31)
g = np.arange(61,103)
h = np.concatenate((f,g))


train_data = NasaDataset(profilelist=h,root_dir=dataset_dir1,
                vza_list1 = vza_list1,vza_list2 = vza_list2, sza_list1 = sza_list1,sza_list2 = sza_list2,
                        patch_size=64,stride=10)

loader = DataLoader(train_data, batch_size=params['batch_size'],shuffle=True)
data_mean, data_std = get_mean_and_std(loader)

transform_func = T.Compose([
T.Normalize(mean=data_mean, std=data_std)
])

del loader
del train_data

test_data = NasaDataset(profilelist=np.arange(31,51),root_dir=dataset_dir1,
                vza_list1 = vza_list1,vza_list2 = vza_list2, sza_list1 = sza_list1,sza_list2 = sza_list2,
                        patch_size=64,stride=10,transform=transform_func,add_dis=True)
dataloader = DataLoader(test_data, batch_size=128,shuffle=False)
total_mse_loss=[]

# Generate image.
netG.eval()

for i, data_batch in enumerate(dataloader, 0):

    r_test, m_test = data_batch['rad_patches'],data_batch['rad_patches2']

    for p_b in range(0, len(r_test)):    

        noise   = r_test[p_b]
        target  = m_test[p_b]
        noise   = noise.to(device,dtype=torch.float32)

        with torch.no_grad():
            generated_img1 = netG(noise).detach().cpu()
        # generated_img1 = generated_img1.numpy()

        mse_loss = torch.mean((generated_img1-target)**2)
        total_mse_loss.append(mse_loss)



print("Test MSE Loss: ",np.mean(total_mse_loss), "Std: ", np.std(total_mse_loss))
print("Done!")