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
from v13_config import params
from v18_dataset import NasaDataset
from visualization import *

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--device', type=int,  default=1, help='CUDA')
    args = parser.parse_args()
    return args


if(params['dataset'] == 'MNIST'):
    from models.mnist_model import Generator, Discriminator, DHead, QHead
elif(params['dataset'] == 'SVHN'):
    from models.svhn_model import Generator, Discriminator, DHead, QHead
elif(params['dataset'] == 'CelebA'):
    from models.celeba_model import Generator, Discriminator, DHead, QHead
elif(params['dataset'] == 'FashionMNIST'):
    from models.mnist_model import Generator, Discriminator, DHead, QHead
elif(params['dataset'] == 'Cloud'):
    # Need to decide new models
    # Consider the input, output size of each blocks
    from models.cloud_model import Generator, Discriminator, DHead, QHead

elif(params['dataset'] == 'Cloud2'):
    # Need to decide new models
    # Consider the input, output size of each blocks
    from models.cloud_model2 import Generator, Discriminator, DHead, QHead

elif(params['dataset'] == 'Cloud3'):
    # Need to decide new models
    # Consider the input, output size of each blocks
    from models.cloud_model3 import Generator, Discriminator, DHead, QHead

elif(params['dataset'] == 'Cloud21'):
    # Need to decide new models
    # Consider the input, output size of each blocks
    from models.cloud_model21 import Generator, Discriminator, DHead, QHead

elif(params['dataset'] == 'Cloud18'):
    # Need to decide new models
    # Consider the input, output size of each blocks
    from models.cloud_model18 import Generator, Discriminator, DHead, QHead

# Set random seed for reproducibility.
seed = 1123
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

# Use GPU if available.
# device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Parse the arguments
args = parse_args()

# Check if the GPU is available
device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")

print(device, " will be used.\n")

# create directory to save checkpoints
saved_model_root_dir = "v13_saved_model/sza_"+str(params['sza_list2'])+"_vza_"+str(params['vza_list2'])
try:
    os.makedirs(saved_model_root_dir)
except FileExistsError:
    print("folder already exists")


# Dataset
f = np.arange(1,31)
g = np.arange(61,103)
h = np.concatenate((f,g))

dataset_dir1 = "/home/local/AD/ztushar1/Data/LES_vers1_multiangle_results"
# sza_list = [60.0,40.0,20.0,4.0]
vza_list1 = params['vza_list1']
vza_list2 = params['vza_list2']
sza_list1 = params['sza_list1']
sza_list2 = params['sza_list2']
profilelist = h
train_data = NasaDataset(profilelist=profilelist,root_dir=dataset_dir1,
                vza_list1 = vza_list1,vza_list2 = vza_list2, sza_list1 = sza_list1,sza_list2 = sza_list2,
                        patch_size=64,stride=10)
print(len(train_data))
loader = DataLoader(train_data, batch_size=params['batch_size'],shuffle=True)
data_mean, data_std = get_mean_and_std(loader)

transform_func = T.Compose([
T.Normalize(mean=data_mean, std=data_std)
])
loader.dataset.set_transform(transform_func)
loader.dataset.set_dis_code(True)

# Set appropriate hyperparameters depending on the dataset used.
# The values given in the InfoGAN paper are used.
# num_z : dimension of incompressible noise.
# num_dis_c : number of discrete latent code used.
# dis_c_dim : dimension of discrete latent code.
# num_con_c : number of continuous latent code used.
if(params['dataset'] == 'MNIST'):
    params['num_z'] = 62
    params['num_dis_c'] = 1
    params['dis_c_dim'] = 10
    params['num_con_c'] = 2

elif(params['dataset'] == 'Cloud18'):
    params['num_z'] = 1
    params['num_dis_c'] = 11
    params['dis_c_dim'] = 72
    params['num_con_c'] = 0
# # Plot the training images.
# sample_batch = next(iter(dataloader))
# plt.figure(figsize=(3, 3))
# plt.axis("off")
# plt.imshow(np.transpose(vutils.make_grid(
#     sample_batch['reflectance'][0].to(device)[ : 9], nrow=3, padding=2, normalize=True).cpu(), (1, 2, 0)))
# plt.savefig('Training Images {}'.format(params['dataset']))
# plt.close('all')

for fold in range(5):
    saved_model_dir = saved_model_root_dir+"/fold_%01d"%(fold)
    try:
        os.makedirs(saved_model_dir)
    except FileExistsError:
        print("folder already exists")
    # Initialise the network.
    netG = Generator(in_channels=13).to(device)
    netG.apply(weights_init)
    # print(netG)

    discriminator = Discriminator(in_channels=2).to(device)
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

    # Fixed Noise
    # z = torch.randn(100, params['num_z'], 1, 1, device=device)
    sample_batch =  loader.dataset[34]
    fixed_noise = sample_batch['rad_patches'][0]
    fixed_noise = torch.unsqueeze(fixed_noise,0)
    fixed_noise = fixed_noise.to(device,dtype=torch.float32)

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

        for i, data_batch in enumerate(loader, 0):
            r_train, m_train = data_batch['rad_patches'],data_batch['rad_patches2']
            idxx_train       = data_batch['idxx']
            print(i)

            temp_G_losses = []
            temp_D_losses = []
            for p_b in range(0, len(r_train)):

                # X_train        = r_train[p_b]
                # Y_train        = m_train[p_b]

                data = m_train[p_b]
                # Get batch size
                b_size = data.size(0)
                # Transfer data tensor to GPU/CPU (device)
                real_data = data.to(device,dtype=torch.float32)

                # Updating discriminator and DHead
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
                # noise, idx = noise_sample(params['num_dis_c'], params['dis_c_dim'], params['num_con_c'], params['num_z'], b_size, device)
                noise, idx = r_train[p_b] , idxx_train[p_b]
                noise = noise.to(device,dtype=torch.float32)
                # print(" probs fake: ", torch.min(noise), torch.max(noise))
                # print("Type of idx: ", type(idx), " Shape of idx: ", idx.shape, " p_b: ", p_b)

                
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

                # Updating Generator and QHead
                optimG.zero_grad()

                # Fake data treated as real.
                output = discriminator(fake_data)
                label.fill_(real_label)
                probs_fake = netD(output).view(-1)
                gen_loss = criterionD(probs_fake, label)

                q_logits, q_mu, q_var = netQ(output)
                target = idx.to(device, dtype=torch.float32)

                

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

                temp_G_losses.append(G_loss.item())
                temp_D_losses.append(D_loss.item())


            #---------------------------------
            # Check progress of training.
            if i != 0 and i%100 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                    % (epoch+1, params['num_epochs'], i, len(dataloader), 
                        np.average(temp_D_losses), np.average(temp_G_losses)))


            # Save the losses for plotting.
            G_losses.append(np.average(temp_G_losses))
            D_losses.append(np.average(temp_D_losses))

            # iters += 1

        epoch_time = time.time() - epoch_start_time
        print("Time taken for Epoch %d: %.2fs" %(epoch + 1, epoch_time))
        # Generate image after each epoch to check performance of the generator. Used for creating animated gif later.
        with torch.no_grad():
            gen_data = netG(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(gen_data, nrow=3, padding=2, normalize=True))

        # Generate image to check performance of generator.
        if((epoch+1) == 1 or (epoch+1) == params['num_epochs']/2):
            with torch.no_grad():
                gen_data = netG(fixed_noise).detach().cpu()
            plt.figure(figsize=(10, 10))
            plt.axis("off")
            # plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True), (1,2,0)))
            fname = "Epoch_%d {}".format(params['dataset']) %(epoch+1)
            plot_cot2(gen_data[0,0],"Radiance at 0.66um",fname,False,[0,2])
            # plt.savefig("Epoch_%d {}".format(params['dataset']) %(epoch+1))
            plt.close('all')

        # Save network weights.
        if (epoch+1) % params['save_epoch'] == 0:
            torch.save({
                'netG' : netG.state_dict(),
                'discriminator' : discriminator.state_dict(),
                'netD' : netD.state_dict(),
                'netQ' : netQ.state_dict(),
                'optimD' : optimD.state_dict(),
                'optimG' : optimG.state_dict(),
                'params' : params
                }, saved_model_dir+'/model_epoch_%d_{}'.format(params['dataset']) %(epoch+1))

    training_time = time.time() - start_time
    print("-"*50)
    print('Training finished!\nTotal Time for Training: %.2fm' %(training_time / 60))
    print("-"*50)

    # Generate image to check performance of trained generator.
    with torch.no_grad():
        gen_data = netG(fixed_noise).detach().cpu()
    plt.figure(figsize=(10, 10))
    # plt.axis("off")
    # plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True), (1,2,0)))
    # plt.savefig("Epoch_%d_{}".format(params['dataset']) %(params['num_epochs']))
    fname = "Epoch_%d_{}".format(params['dataset']) %(params['num_epochs'])
    plot_cot2(gen_data[0,0],"Radiance at 0.66um",fname,False,[0,2])


    # Save network weights.
    torch.save({
        'netG' : netG.state_dict(),
        'discriminator' : discriminator.state_dict(),
        'netD' : netD.state_dict(),
        'netQ' : netQ.state_dict(),
        'optimD' : optimD.state_dict(),
        'optimG' : optimG.state_dict(),
        'params' : params,
        'training_time':training_time
        }, 'checkpoint/model_final_{}'.format(params['dataset']))


    # Plot the training losses.
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("Loss Curve {}".format(params['dataset']))

    # # Animation showing the improvements of the generator.
    # fig = plt.figure(figsize=(10,10))
    # plt.axis("off")
    # ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    # anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    # anim.save('infoGAN_{}.gif'.format(params['dataset']), dpi=80, writer='imagemagick')
    # plt.show()