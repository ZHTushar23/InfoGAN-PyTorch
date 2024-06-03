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
from v18_dataset import NasaDataset
from v17_config import params

# create directory to save checkpoints
saved_model_root_dir = "v17_saved_model/sza_"+str(params['sza_list2'])+"_vza_"+str(params['vza_list2'])



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
# test_data = NasaDataset(profilelist=np.arange(31,51),root_dir=dataset_dir1,
#                 vza_list1 = vza_list1,vza_list2 = vza_list2, sza_list1 = [60.0],sza_list2 = [4.0],
#                         patch_size=64,stride=10,transform=transform_func,add_dis=True)

dataloader = DataLoader(test_data, batch_size=128,shuffle=False)





# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Get the 'params' dictionary from the loaded state_dict.
# model_params = state_dict['params']

# Create the generator network.
netG = Generator(13).to(device)

cv_mse_loss = []

for fold in range (5):
    # if fold==3:
    #     continue
    total_mse_loss=[]
    saved_model_dir = saved_model_root_dir+"/nfold_%01d"%(fold)
    load_path = saved_model_dir+'/model_epoch_%d_{}'.format(params['dataset']) %(350)
    # load_path  = 'checkpoint/model_final_{}'.format(params['dataset'])
    # Load the checkpoint file
    state_dict = torch.load(load_path)
    # Load the trained generator weights.
    netG.load_state_dict(state_dict['netG'])
    # print(netG)
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


    cv_mse_loss.append(np.mean(total_mse_loss))
    print("Fold MSE Loss: ",np.mean(total_mse_loss), "Std: ", np.std(total_mse_loss))

print("Test MSE Loss: ",np.mean(cv_mse_loss), "Std: ", np.std(cv_mse_loss))
    
print("Done!")