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
from ES2_config_dc import params
from ES2_dataset_dc import NasaDataset
from visualization import *
from pytorchtools import EarlyStopping


if(params['dataset'] == 'MNIST'):
    from models.mnist_model import Generator, Discriminator, DHead, QHead
elif(params['dataset'] == 'Cloud18'):
    # Need to decide new models
    # Consider the input, output size of each blocks
    from models.cloud_model18 import Generator, Discriminator, DHead, QHead
elif(params['dataset'] == 'ES5' or params['dataset'] == 'ES1'):
    # Need to decide new models
    # Consider the input, output size of each blocks
    from models.es5_model import Generator, Discriminator, DHead, QHead

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
saved_model_root_dir = "dc_es2_saved_model"
try:
    os.makedirs(saved_model_root_dir)
except FileExistsError:
    print("folder already exists")


# Dataset
# root_data_dir ="/home/local/AD/ztushar1/multi-view-cot-retrieval/LES_MultiView_100m_64/"
root_data_dir ="/home/ztushar1/psanjay_user/multi-view-cot-retrieval/LES_MultiView_100m_64/"
data_split_dir = "/home/ztushar1/psanjay_user/multi-view-cot-retrieval/"
data_split_sub_dir = "data_split/cot_sza_all_vza_0/"

data_mean, data_std = torch.tensor([0.1357, 0.0980], dtype=torch.float64), torch.tensor([0.1793, 0.0953], dtype=torch.float64)
transform_func = T.Compose([T.Normalize(mean=data_mean, std=data_std)])

train_data = NasaDataset(root_dir=root_data_dir,
            filelist=data_split_dir+data_split_sub_dir+"/train.csv",
            mode="infoGAN_cot_dc",transform=transform_func)
valid_data = NasaDataset(root_dir=root_data_dir,
            filelist=data_split_dir+data_split_sub_dir+"/val.csv",
            mode="infoGAN_cot_dc",transform=transform_func)
test_data = NasaDataset(root_dir=root_data_dir,
            filelist=data_split_dir+data_split_sub_dir+"/test.csv",
            mode="infoGAN_cot_dc",transform=transform_func)

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
    early_stopping = EarlyStopping(patience=200, verbose=True,path=saved_model_dir)


    # Initialise the network.
    netG = Generator(in_channels=13,out_channels=1).to(device)
    netG.apply(weights_init)
    # print(netG)

    discriminator = Discriminator(in_channels=1).to(device)
    discriminator.apply(weights_init)
    # print(discriminator)

    netD = DHead().to(device)
    netD.apply(weights_init)
    # print(netD)

    netQ = QHead().to(device)
    netQ.apply(weights_init)
    # print(netQ)

    # Loss for discrimination between real and fake images.
    criterionD = nn.BCELoss()
    # Loss for discrete latent code.
    criterionQ_dis = nn.CrossEntropyLoss()
    # Loss for continuous latent code.
    criterionQ_con = NormalNLLLoss()

    # Adam optimiser is used.
    optimD = optim.Adam([{'params': discriminator.parameters()}, {'params': netD.parameters()}], lr=params['learning_rate'], betas=(params['beta1'], params['beta2']))
    optimG = optim.Adam([{'params': netG.parameters()}, {'params': netQ.parameters()}], lr=params['learning_rate'], betas=(params['beta1'], params['beta2']))

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
            noise, real_data = data_batch['input'],data_batch['target']
            idxx, style_code = data_batch['idxx'], data_batch['code']

            # Get batch size
            b_size = real_data.size(0)
            # Transfer data tensor to GPU/CPU (device)
            real_data = real_data.to(device,dtype=torch.float32)

            #.1 Updating discriminator and DHead
            optimD.zero_grad()
            # Real data
            label = torch.full((b_size, ), real_label, device=device).to(torch.float32)
            output1 = discriminator(real_data)
            probs_real = netD(output1).view(-1)
            loss_real = criterionD(probs_real, label)
            # Calculate gradients.
            loss_real.backward()

            # Fake data
            label.fill_(fake_label)
            noise = noise.to(device,dtype=torch.float32)
            style_code = style_code.to(device,dtype=torch.float32)

            # fake_data = netG(noise,style_code)     
            fake_data = netG(noise)        
            output2 = discriminator(fake_data.detach())            
            probs_fake = netD(output2).view(-1)
            
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
            output = discriminator(fake_data)
            label.fill_(real_label)
            probs_fake = netD(output).view(-1)
            gen_loss = criterionD(probs_fake, label)

            q_logits, q_mu, q_var = netQ(output)
            target = idxx.to(device, dtype=torch.float32)

            # Calculating loss for discrete latent code.
            dis_loss1 = criterionQ_dis(q_logits[:,:4],target[:,:4].softmax(dim=1))
            dis_loss2 = criterionQ_dis(q_logits[:,4:],target[:,4:].softmax(dim=1))
            dis_loss = dis_loss1+dis_loss2
            # dis_loss = 0
            # for j in range(params['num_dis_c']):
            #     dis_loss += criterionQ_dis(q_logits[:, j], target[:,j])

            # # Calculating loss for continuous latent code.
            con_loss = 0
            # if (params['num_con_c'] != 0):
            #     con_loss = criterionQ_con(noise[:, params['num_z']+ params['num_dis_c']*params['dis_c_dim'] : ].view(-1, params['num_con_c']), q_mu, q_var)*0.1

            # Net loss for generator.
            G_loss = gen_loss + dis_loss + con_loss
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
            noise, target, style_code = data_batch['input'],data_batch['target'],data_batch['code']
            noise   = noise.to(device,dtype=torch.float32)
            style_code = style_code.to(device,dtype=torch.float32)

            with torch.no_grad():
                # generated_img1 = netG(noise,style_code).detach().cpu() 
                generated_img1 = netG(noise).detach().cpu()            
            mse_loss = torch.mean((generated_img1-target)**2)
            t_valid_loss.append(mse_loss)

        valid_loss=np.average(t_valid_loss)
        # Check progress of training.
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tVal_Loss_G: %.4f'
            % (epoch+1, params['num_epochs'], i, len(loader), 
                np.average(temp_D_losses), np.average(temp_G_losses),valid_loss))

        # Early Stopping                
        early_stopping(val_loss = valid_loss, netD = netD,
                        netG = netG, netQ = netQ, 
                        discriminator = discriminator,
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

