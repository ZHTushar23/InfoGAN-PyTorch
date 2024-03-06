import argparse

import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', required=True, help='Checkpoint to load path from')
args = parser.parse_args()

from models.cloud_model4 import Generator
from utils import *
from visualization import *
from v4_dataset import NasaDataset

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

dataset_dir ="/home/local/AD/ztushar1/LES102_MultiView_100m_F2/"
dataset = NasaDataset(root_dir=dataset_dir,mode="test")
dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=1, 
                                            shuffle=False)

total_mse_loss=[]
for i, sample_batch in enumerate(dataloader, 0):
    noise1 = sample_batch['CWN']
    noise1 = noise1.to(device,dtype=torch.float32)
    cf1 = sample_batch['sza_emb']
    cf1 = cf1.to(device,dtype=torch.float32)
    cf2 = sample_batch['vza_emb']
    cf2 = cf2.to(device,dtype=torch.float32)    

    # Generate image.
    netG.eval()
    with torch.no_grad():
        generated_img1 = netG(noise1,cf1,cf2).detach().cpu()
    generated_img1 = generated_img1.numpy()

    sza,vza,patch_name = sample_batch['sza'],sample_batch['vza'], sample_batch['name']
    patch_name=patch_name[0]
    sza = sza.numpy()
    vza = vza.numpy()

    dir_name = "v4_cloud"
    # fname = dir_name+"/rad066_pred_"+patch_name+"_SZA_%02d_VZA_%02d.png"%(sza,vza)
    # plot_cot2(generated_img1[0,0],"Pred Radiance at 0.66um",fname,False,[0,2])

    r_data = sample_batch['reflectance'].numpy()
    # fname = dir_name+"/rad066_"+patch_name+"_SZA_%02d_VZA_%02d.png"%(sza,vza)
    # plot_cot2(r_data[0,0],"Radiance at 0.66um",fname,False,[0,2])


    # c_data = sample_batch['cot'].numpy()
    # fname = dir_name+"/cot_"+patch_name+"_SZA_%02d_VZA_%02d.png"%(sza,vza)
    # plot_cot2(np.log(c_data[0,0]+1),"True COT",fname,False,[0,7])

    mse_loss = np.mean((generated_img1[0,0]-r_data[0,0])**2)
    total_mse_loss.append(mse_loss)



print("Test MSE Loss: ",np.mean(total_mse_loss), "Std: ", np.std(total_mse_loss))
print("Done!")