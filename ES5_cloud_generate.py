import argparse

import torch
import torchvision.utils as vutils
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt

from models.es5_model import Generator
from v2_utils import *
from visualization import *
from ES5_dataset import NasaDataset
from ES5_config import params

# create directory to save checkpoints
saved_model_root_dir = "es5_saved_model"



# noise1 , _ = noise_sample(10,10,0,128,9,device)
dataset_dir1 = "/nfs/rs/psanjay/users/ztushar1/LES_vers1_multiangle_results"
# dataset_dir1 = "/home/local/AD/ztushar1/Data/LES_vers1_multiangle_results"
# root_data_dir ="/home/local/AD/ztushar1/multi-view-cot-retrieval/LES_MultiView_100m_64/"
root_data_dir ="/home/ztushar1/psanjay_user/multi-view-cot-retrieval/LES_MultiView_100m_64/"
data_split_dir = "/home/ztushar1/psanjay_user/multi-view-cot-retrieval/"


data_mean, data_std = torch.tensor([0.1600, 0.1121], dtype=torch.float64), torch.tensor([0.2106, 0.1189], dtype=torch.float64)
transform_func = T.Compose([
T.Normalize(mean=data_mean, std=data_std)
])

test_data = NasaDataset(root_dir=root_data_dir,
            filelist=data_split_dir+"data_split/cot_all/test.csv",
            mode="infoGAN_cot",transform=transform_func)

dataloader = DataLoader(test_data, batch_size=params['batch_size'],shuffle=False)


# Set the device to run on: GPU or CPU.
device = torch.device("cuda" if(torch.cuda.is_available()) else "cpu")
# Get the 'params' dictionary from the loaded state_dict.
# model_params = state_dict['params']

# Create the generator network.
netG = Generator(in_ch=2,out_ch=1).to(device)

cv_mse_loss = []

for fold in range (1):
    total_mse_loss=[]
    saved_model_dir = saved_model_root_dir+"/fold_%01d"%(fold)
    load_path = saved_model_dir+'/model_final_{}'.format(params['dataset'])

    # Load the checkpoint file
    state_dict = torch.load(load_path)
    # Load the trained generator weights.
    netG.load_state_dict(state_dict['netG'])
    # print(netG)
    # Generate image.
    netG.eval()

    for i, data_batch in enumerate(dataloader, 0):

        noise, real_data = data_batch['input'],data_batch['target']
        idxx, style_code = data_batch['idxx'], data_batch['code']
        noise   = noise.to(device,dtype=torch.float32)
        style_code = style_code.to(device,dtype=torch.float32)

        with torch.no_grad():
            generated_img1 = netG(noise,style_code).detach().cpu()
            # generated_img1 = generated_img1.numpy()

        mse_loss = torch.mean((generated_img1-real_data)**2)
        total_mse_loss.append(mse_loss)


    cv_mse_loss.append(np.mean(total_mse_loss))
    print("Fold MSE Loss: ",np.mean(total_mse_loss), "Std: ", np.std(total_mse_loss))

print("Test MSE Loss: ",np.mean(cv_mse_loss), "Std: ", np.std(cv_mse_loss))
    
print("Done!")