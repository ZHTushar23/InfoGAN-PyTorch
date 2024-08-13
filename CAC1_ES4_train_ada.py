import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import random
import argparse
import os


# from dataloader import get_data
from v2_utils import *
from CAC1_ES4_config import params
from CAC1_ES4_dataset import NasaDataset
from visualization import *
from pytorchtools import EarlyStopping2
from models.CAC_model import Generator, Discriminator



# Set random seed for reproducibility.
seed = 1123
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

# Use GPU if available.
# device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Parse the arguments
# Check if the GPU is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
print(f'Main Selected device: {device}')

# create directory to save checkpoints
saved_model_root_dir = "CAC1_ES4_saved_model"
try:
    os.makedirs(saved_model_root_dir)
except FileExistsError:
    print("folder already exists")


# Dataset
# root_data_dir ="/home/local/AD/ztushar1/multi-view-cot-retrieval/LES_MultiView_100m_64/"
root_data_dir ="/home/ztushar1/psanjay_user/multi-view-cot-retrieval/LES_MultiView_100m_64/"
data_split_dir = "/home/ztushar1/psanjay_user/multi-view-cot-retrieval/"
data_split_sub_dir = "data_split/cot_sza_20_vza_all/"

data_mean, data_std = torch.tensor([0.1410, 0.1020], dtype=torch.float64), torch.tensor([0.1596, 0.0854], dtype=torch.float64)
transform_func = T.Compose([T.Normalize(mean=data_mean, std=data_std)])

train_data = NasaDataset(root_dir=root_data_dir,
            filelist=data_split_dir+data_split_sub_dir+"/train.csv",
            mode="cot",transform=transform_func)
valid_data = NasaDataset(root_dir=root_data_dir,
            filelist=data_split_dir+data_split_sub_dir+"/val.csv",
            mode="cot",transform=transform_func)
test_data = NasaDataset(root_dir=root_data_dir,
            filelist=data_split_dir+data_split_sub_dir+"/test.csv",
            mode="cot",transform=transform_func)

loader = DataLoader(train_data, batch_size=params['batch_size'],shuffle=True)
val_loader = DataLoader(valid_data, batch_size=params['batch_size'],shuffle=False)
test_loader = DataLoader(test_data, batch_size=params['batch_size'],shuffle=False)


for fold in range(2):
    saved_model_dir = saved_model_root_dir+"/fold_%01d"%(fold)
    try:
        os.makedirs(saved_model_dir)
    except FileExistsError:
        print("folder already exists")
    # initialize the early_stopping object
    early_stopping = EarlyStopping2(patience=200, verbose=True,path=saved_model_dir)


    # Initialise the network.
    netG = Generator(in_ch=2,out_ch=1).to(device)
    netG.apply(weights_init)
    # print(netG)

    netD = Discriminator(img_size=64, in_ch=1).to(device)
    netD.apply(weights_init)


    # Loss for discrimination between real and fake images.
    criterionD = nn.BCELoss()

    # Adam optimiser is used.
    optimD = optim.Adam([{'params': netD.parameters()}], lr=params['learning_rate'], betas=(params['beta1'], params['beta2']))
    optimG = optim.Adam([{'params': netG.parameters()}], lr=params['learning_rate'], betas=(params['beta1'], params['beta2']))

    real_label = 1
    fake_label = 0

    # List variables to store results pf training.
    img_list = []
    G_losses = []
    D_losses = []

    print("-"*25)
    print("Starting Training Loop...\n")
    print('Epochs: %d\nDataset: {}\nBatch Size: %d\nLength of Data Loader: %d'.format(params['dataset']) % (params['num_epochs'], params['batch_size'], len(loader)))
    print("-"*25)

    start_time = time.time()
    # iters = 0

    for epoch in range(params['num_epochs']):
        epoch_start_time = time.time()

        temp_G_losses = []
        temp_D_losses = []


        for i, data_batch in enumerate(loader, 0):
            input_img1, angle_code1 = data_batch['input1'],data_batch['code1']
            input_img2, angle_code2 = data_batch['input2'],data_batch['code2']
            real_data = data_batch['target']

            # Get batch size
            b_size = real_data.size(0)
            # Transfer data tensor to GPU/CPU (device)
            real_data = real_data.to(device,dtype=torch.float32)

            #.1 Updating discriminator and DHead
            optimD.zero_grad()
            # Real data
            label = torch.full((b_size, ), real_label, device=device).to(torch.float32)
            probs_real = netD(real_data).view(-1)
            loss_real = criterionD(probs_real, label)
            # Calculate gradients.
            loss_real.backward()

            # Fake data
            label.fill_(fake_label)
            input_img1 = input_img1.to(device,dtype=torch.float32)
            angle_code1 = angle_code1.to(device,dtype=torch.float32)

            fake_data = netG(input_img1, angle_code1)                     
            probs_fake = netD(fake_data.detach()).view(-1)
            
            loss_fake = criterionD(probs_fake, label)
            # Calculate gradients.
            loss_fake.backward()

            # Net Loss for the discriminator
            D_loss = loss_real + loss_fake
            # Update parameters
            optimD.step()



            #.2 Updating Generator and QHead
            optimG.zero_grad()

            # Fake data treated as real.
            label.fill_(real_label)
            probs_fake = netD(fake_data).view(-1)

            # Adv Loss
            # print("Hello",label.shape, probs_fake.shape)
            # print(probs_fake.min(), probs_fake.max())
            gen_loss = criterionD(probs_fake, label)

            # Property Consistency loss
            input_img2 = input_img2.to(device,dtype=torch.float32)
            angle_code2 = angle_code2.to(device,dtype=torch.float32)
            fake_data2 = netG(input_img2, angle_code2)
            fake_data2 = fake_data2.detach()
            loss_ds = torch.mean(torch.abs(fake_data - fake_data2))

            # Net loss for generator.
            G_loss = gen_loss + loss_ds
            # Calculate gradients.
            G_loss.backward()
            # Update parameters.
            optimG.step()

            # Save the losses for plotting.
            temp_G_losses.append(G_loss.item())
            temp_D_losses.append(D_loss.item())



        G_losses.append( np.average(temp_G_losses))
        D_losses.append( np.average(temp_D_losses))


        epoch_time = time.time() - epoch_start_time
        print("Time taken for Epoch %d: %.2fs" %(epoch + 1, epoch_time))

        # validate the model #
        ######################
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        t_valid_loss=[]
        for i, data_batch in enumerate(val_loader):
            input_img1, angle_code1 = data_batch['input1'],data_batch['code1']
            real_data = data_batch['target']
            input_img1 = input_img1.to(device,dtype=torch.float32)
            angle_code1 = angle_code1.to(device,dtype=torch.float32)


            with torch.no_grad():
                generated_img1 = netG(input_img1,angle_code1).detach().cpu()            
            mse_loss = torch.mean((generated_img1-real_data)**2)
            t_valid_loss.append(mse_loss)

        valid_loss=np.average(t_valid_loss)
        # Check progress of training.
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tVal_Loss_G: %.4f'
            % (epoch+1, params['num_epochs'], i, len(loader), 
                np.average(temp_D_losses), np.average(temp_G_losses),valid_loss))

        # Early Stopping                
        early_stopping(val_loss = valid_loss, netD = netD,
                        netG = netG,
                        optimD = optimD, optimG = optimG,
                        params= params)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # # Save network weights.
        # if (epoch+1) % params['save_epoch'] == 0:
        #     torch.save({
        #         'netG' : netG.state_dict(),
        #         'discriminator' : discriminator.state_dict(),
        #         'netD' : netD.state_dict(),
        #         'netQ' : netQ.state_dict(),
        #         'optimD' : optimD.state_dict(),
        #         'optimG' : optimG.state_dict(),
        #         'params' : params
        #         }, saved_model_dir+'/model_epoch_%d_{}'.format(params['dataset']) %(epoch+1))

    training_time = time.time() - start_time
    print("-"*50)
    print('Training finished!\nTotal Time for Training: %.2fm' %(training_time / 60))
    print("-"*50)


    # # Save network weights.
    # torch.save({
    #     'netG' : netG.state_dict(),
    #     'discriminator' : discriminator.state_dict(),
    #     'netD' : netD.state_dict(),
    #     'netQ' : netQ.state_dict(),
    #     'optimD' : optimD.state_dict(),
    #     'optimG' : optimG.state_dict(),
    #     'params' : params,
    #     'training_time':training_time
    #     }, saved_model_dir+'/model_final_{}'.format(params['dataset']))


    # Plot the training losses.
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(saved_model_dir+"/Loss Curve {}".format(params['dataset']))

