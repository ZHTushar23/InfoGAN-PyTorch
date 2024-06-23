import argparse

import torch
import torchvision.utils as vutils
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt

from models.cloud_model18 import Generator
from v2_utils import *
from visualization import *
from v29_dataset import NasaDataset
from v29_config import params

# create directory to save checkpoints
saved_model_root_dir = "v29_saved_model/sza_"+str(params['sza_list2'])+"_vza_"+str(params['vza_list2'])



# noise1 , _ = noise_sample(10,10,0,128,9,device)
dataset_dir1 = "/nfs/rs/psanjay/users/ztushar1/LES_vers1_multiangle_results"
# dataset_dir1 = "/home/local/AD/ztushar1/Data/LES_vers1_multiangle_results"
# root_data_dir ="/home/local/AD/ztushar1/multi-view-cot-retrieval/LES_MultiView_100m_64/"
root_data_dir ="/home/ztushar1/psanjay_user/multi-view-cot-retrieval/LES_MultiView_100m_64/"

data_mean, data_std = torch.tensor([0.1329, 0.1087], dtype=torch.float64), torch.tensor([0.1537, 0.0924], dtype=torch.float64)
transform_func = T.Compose([
T.Normalize(mean=data_mean, std=data_std)
])

test_data = NasaDataset(root_dir=root_data_dir,
            filelist="data_split/nadir_to_all/test.csv",
            mode="infoGAN",transform=transform_func)
# SZA only
# test_data = NasaDataset(root_dir=root_data_dir,
#             filelist="data_split/nadir_to_all/test_sza_20_vza_15.csv",
#             mode="infoGAN",transform=transform_func)

dataloader = DataLoader(test_data, batch_size=128,shuffle=False)


# Set the device to run on: GPU or CPU.
device = torch.device("cuda" if(torch.cuda.is_available()) else "cpu")
# Get the 'params' dictionary from the loaded state_dict.
# model_params = state_dict['params']

# Create the generator network.
netG = Generator(3).to(device)

cv_mse_loss = []

for fold in range (2):
    total_mse_loss=[]
    saved_model_dir = saved_model_root_dir+"/mfold_%01d"%(fold)
    load_path = saved_model_dir+'/model_final_{}'.format(params['dataset'])

    # Load the checkpoint file
    state_dict = torch.load(load_path)
    # Load the trained generator weights.
    netG.load_state_dict(state_dict['netG'])
    # print(netG)
    # Generate image.
    netG.eval()

    for i, data_batch in enumerate(dataloader, 0):

        noise, target = data_batch['input'],data_batch['target']

        noise   = noise.to(device,dtype=torch.float32)

        with torch.no_grad():
            generated_img1 = netG(noise).detach().cpu()
        # generated_img1 = generated_img1.numpy()

        mse_loss = torch.mean((generated_img1-target)**2)
        total_mse_loss.append(mse_loss)


    cv_mse_loss.append(np.mean(total_mse_loss))
    print("Fold MSE Loss: ",np.mean(total_mse_loss), "Std: ", np.std(total_mse_loss))

print("Test MSE Loss: ",np.mean(cv_mse_loss), "Std: ", np.std(cv_mse_loss))
    
print("Done!")