import argparse

import torch
import torchvision.utils as vutils
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt

from models.cloud_model18 import *
from v2_utils import *
from visualization import *
from v29_dataset import NasaDataset
from v29_config import params




def compute_accuracy(predictions, true_labels):
    # # Ensure predictions and true_labels are torch tensors
    # predictions = torch.tensor(predictions)
    # true_labels = torch.tensor(true_labels)
    
    # Compare each row of predictions with the corresponding row of true labels
    correct_rows = torch.all(predictions == true_labels, dim=1)
    
    # Calculate the number of completely correct rows
    num_correct_rows = torch.sum(correct_rows)
    
    # Calculate accuracy
    accuracy = num_correct_rows.item() / predictions.size(0)
    
    return accuracy


# create directory to save checkpoints
saved_model_root_dir = "v29_saved_model/sza_"+str(params['sza_list2'])+"_vza_"+str(params['vza_list2'])



# noise1 , _ = noise_sample(10,10,0,128,9,device)
dataset_dir1 = "/nfs/rs/psanjay/users/ztushar1/LES_vers1_multiangle_results"
# dataset_dir1 = "/home/local/AD/ztushar1/Data/LES_vers1_multiangle_results"
# root_data_dir ="/home/local/AD/ztushar1/multi-view-cot-retrieval/LES_MultiView_100m_64/"
root_data_dir ="/home/ztushar1/psanjay_user/multi-view-cot-retrieval/LES_MultiView_100m_64/"

# sza_list = [60.0,40.0,20.0,4.0]
vza_list1 = params['vza_list1']
vza_list2 = params['vza_list2']
sza_list1 = params['sza_list1']
sza_list2 = params['sza_list2']

f = np.arange(1,31)
g = np.arange(61,103)
h = np.concatenate((f,g))


# train_data = NasaDataset(profilelist=h,root_dir=dataset_dir1,
#                 vza_list1 = vza_list1,vza_list2 = vza_list2, sza_list1 = sza_list1,sza_list2 = sza_list2,
#                         patch_size=64,stride=10)

# loader = DataLoader(train_data, batch_size=params['batch_size'],shuffle=True)
# data_mean, data_std = get_mean_and_std(loader)

# print(data_mean,data_std)
# transform_func = T.Compose([
# T.Normalize(mean=data_mean, std=data_std)
# ])

# del loader
# del train_data
data_mean, data_std = torch.tensor([0.1329, 0.1087], dtype=torch.float64), torch.tensor([0.1537, 0.0924], dtype=torch.float64)
transform_func = T.Compose([
T.Normalize(mean=data_mean, std=data_std)
])

test_data = NasaDataset(root_dir=root_data_dir,
            filelist="data_split/nadir_to_all/test.csv",
            mode="infoGAN",transform=transform_func)
# test_loader = DataLoader(test_data, batch_size=params['batch_size'],shuffle=False)


# test_data = NasaDataset(profilelist=np.arange(31,51),root_dir=dataset_dir1,
#                 vza_list1 = vza_list1,vza_list2 = vza_list2, sza_list1 = sza_list1,sza_list2 = sza_list2,
#                         patch_size=64,stride=10,transform=transform_func,add_dis=True)
# test_data = NasaDataset(profilelist=np.arange(31,51),root_dir=dataset_dir1,
#                 vza_list1 = [0],vza_list2 = [60], sza_list1 = [4.0],sza_list2 = [60.0],
#                         patch_size=64,stride=10,transform=transform_func,add_dis=True)

dataloader = DataLoader(test_data, batch_size=128,shuffle=False)


# Set the device to run on: GPU or CPU.
device = torch.device("cuda" if(torch.cuda.is_available()) else "cpu")
# Get the 'params' dictionary from the loaded state_dict.
# model_params = state_dict['params']

# Create the generator network.
netG = Generator(3).to(device)
netQ = QHead().to(device)
discriminator = Discriminator(in_channels=2).to(device)
netD = DHead().to(device)


cv_mse_loss = []
cv_sz_ac , cv_vz_ac      = [],[]

for fold in range (2):
    total_mse_loss=[]
    sz_ac , vz_ac      = [],[]
    saved_model_dir = saved_model_root_dir+"/mfold_%01d"%(fold)
    load_path = saved_model_dir+'/model_final_{}'.format(params['dataset'])

    # Load the checkpoint file
    state_dict = torch.load(load_path)
    # Load the trained generator weights.
    netG.load_state_dict(state_dict['netG'])
    netQ.load_state_dict(state_dict['netQ'])
    discriminator.load_state_dict(state_dict['discriminator'])
    netD.load_state_dict(state_dict['netD'])
    # print(netG)
    # Generate image.
    netG.eval()
    netQ.eval()
    discriminator.eval()
    netD.eval()

    for i, data_batch in enumerate(dataloader, 0):

        noise, target, true_labels = data_batch['input'],data_batch['target'], data_batch['idxx']

        noise   = noise.to(device,dtype=torch.float32)
        with torch.no_grad():
            generated_img1     = netG(noise)
            outputt            = discriminator(generated_img1)
            true_fake          = netD(outputt)
            pred_label, _, _   = netQ(outputt)
            generated_img1     = generated_img1.detach().cpu()
        # generated_img1 = generated_img1.numpy()
        pred_label = pred_label.detach().cpu()

        mse_loss = torch.mean((generated_img1-target)**2)
        total_mse_loss.append(mse_loss)
        # print(pred_label.shape, true_fake.shape)
        sza_acc = compute_accuracy(pred_label[:,:4]>0.5,true_labels[:,:4])
        vza_acc = compute_accuracy(pred_label[:,4:]>0.5,true_labels[:,4:])

        sz_ac.append(sza_acc)
        vz_ac.append(vza_acc)

            # print(f"SZA Accuracy: {sza_acc * 100:.2f}%")
            # print(f"VZA Accuracy: {vza_acc * 100:.2f}%")


    cv_mse_loss.append(np.mean(total_mse_loss))
    cv_sz_ac.append(np.mean(sz_ac))
    cv_vz_ac.append(np.mean(vz_ac))
    print("Fold MSE Loss: ",np.mean(total_mse_loss), "Std: ", np.std(total_mse_loss))
    print(f"SZA Accuracy: {np.mean(sz_ac) * 100:.2f}%")
    print(f"VZA Accuracy: {np.mean(vz_ac) * 100:.2f}%")
print("Test MSE Loss: ",np.mean(cv_mse_loss), "Std: ", np.std(cv_mse_loss))
print("Test SZA Acc: ",np.mean(cv_sz_ac)*100, "Std: ", np.std(cv_sz_ac*100))
print("Test VZA Acc: ",np.mean(cv_vz_ac)*100, "Std: ", np.std(cv_vz_ac*100))
    
print("Done!")