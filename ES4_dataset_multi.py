
'''
    Author: Zahid Hassan Tushar
    email: ztushar1@umbc.edu
'''


# Import Libraries
import os
import h5py
from torch.utils.data import Dataset, random_split, DataLoader
import torch
import numpy as np
import torchvision.transforms as T
torch.manual_seed(0)

sza_full_list = [60.0,40.0,20.0,4.0]
vza_full_list = [60,30,15,0,-15,-30,-60]

def _read_csv(infile):
    import csv
    file = open(infile, "r")
    data = list(csv.reader(file, delimiter=","))
    file.close()
    return data[1:]


class NasaDataset(Dataset):
    """  
    Designed for 64x64 patch size rad2rad and cot dataset. 
    Args:
    root_dir = string with numpy patch locations
    filelist = csv file with names of patches. 1st col: input, 2nd col: output/target
    filelist are created using v64_create_data_partition.py supplying profile numbers, sza and vza information.
    """

    def __init__(self, root_dir=None,filelist=None, transform=None, mode='cot'):
        self.root_dir    = root_dir
        self.filelist    = _read_csv(filelist)
        self.transform1  = T.Compose([T.ToTensor()])
        self.transform   = transform
        self.mode        = mode
        if self.mode=='single':
            self.samples, self.targets = self._make_dataset()

    def _make_dataset(self):
        fnames, labels = [], []
        for idx in range(len(self.filelist)):
            cls_fnames = os.path.join(self.root_dir,self.filelist[idx][0])
            fnames += [cls_fnames]
            s = sza_full_list.index(int(float(self.filelist[idx][2])))
            v = vza_full_list.index(int(self.filelist[idx][3]))
            labels += [s*7+v] # assigning class labels based on domain. first domain has label 0, next is 1.
        return fnames, labels


    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # print(self.filelist[idx])

        input_img  = np.load(os.path.join(self.root_dir,self.filelist[idx][0]))
        output_img =None

        # based on the mode, select the target [label/ cot/ radiance/ radiance+discrete code]
        if self.mode=='single':
            output_img = torch.tensor(0)
        elif self.mode =='rad2radm':
            output_img = np.load(os.path.join(self.root_dir,self.filelist[idx][5]))
        else:
            output_img = np.load(os.path.join(self.root_dir,self.filelist[idx][1]))


        if self.mode=='infoGAN_cot' or self.mode=='rad2radm':
            sza = np.float32(self.filelist[idx][3]) # SZA value
            vza = np.float32(self.filelist[idx][4]) # VZA value

            #find index
            s = sza_full_list.index(sza)
            v = vza_full_list.index(vza)

            sza_bits = format(s, '02b')
            vza_bits = format(v, '03b')

            # Concatenate the bits to form the 5-bit code
            dis_c = sza_bits + vza_bits

            # Convert dis_c to a list of integers (0 or 1)
            dis_c_list = [int(bit) for bit in dis_c]

            # Convert the list to a torch tensor
            dis_c_tensor = torch.tensor(dis_c_list, dtype=torch.float32)

            # discrete labels
            idxx = np.zeros((11)) # ['num_dis_c'] = 11, Batch size
            # idxx = torch.zeros((11)) # ['num_dis_c'] = 11, Batch size
            idxx[s]=1
            idxx[v+4]=1

        # Convert to tensor
        input_img =  self.transform1(input_img)
        if self.mode=="cot" or self.mode=="infoGAN_cot":
            output_img = np.log(output_img[:,:,0]+1)
            # print(output_img.shape, " should be 1,64,64")
        
        if self.mode!="single":            
            output_img =  self.transform1(output_img)

        if self.transform:
            input_img =  self.transform(input_img)      
        if self.mode == "infoGAN" or self.mode=="infoGAN_cot" or self.mode=="rad2radm":
            # input_img = torch.cat((input_img, dis_c), dim=0)
            sample = {'input':input_img,'target':output_img,'idxx':idxx,'code':dis_c_tensor}
        else:
            sample = {'input':input_img,'target':output_img}

        
        return sample

def get_mean_and_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for batch, sample in enumerate (loader):
        data = sample['input']
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

if __name__=="__main__":

    root_data_dir ="/home/ztushar1/psanjay_user/multi-view-cot-retrieval/LES_MultiView_100m_64/"
    data_mean, data_std = torch.tensor([0.1433, 0.1034], dtype=torch.float64), torch.tensor([0.1604, 0.0862], dtype=torch.float64)
    transform_func = T.Compose([
    T.Normalize(mean=data_mean, std=data_std)
    ])
    data_split_dir = "/home/ztushar1/psanjay_user/multi-view-cot-retrieval/"
    for fold in range(1):    
        # fold=0
        print("checking fold: ", fold)
        train_data = NasaDataset(root_dir=root_data_dir,filelist=data_split_dir+"data_split/mcot_sza_20_vza_all/train.csv",mode="rad2radm")
        print(len(train_data))
        loader = DataLoader(train_data, batch_size=2,shuffle=False)
        
        temp= []
        temp1= []   
        for i in range(len(loader.dataset)):
            data = loader.dataset[i]
            # get the data
            X, Z, codee = data['input'], data['target'], data['code']
            print(X.shape, Z.shape, codee.shape)
            break
        for i, data in enumerate(loader):
            # get the data
            X, Z, CC = data['input'], data['target'], data['code']
            print(X.shape, Z.shape, CC.shape)
            break
    data_mean, data_std = torch.tensor([0.1433, 0.1034], dtype=torch.float64), torch.tensor([0.1604, 0.0862], dtype=torch.float64)
    transform_func = T.Compose([T.Normalize(mean=data_mean, std=data_std)])
        
    train_data = NasaDataset(root_dir=root_data_dir,
    filelist=data_split_dir+"data_split/mcot_sza_20_vza_all/train.csv",
    mode="rad2radm", transform=transform_func)
    print(len(train_data))
    loader = DataLoader(train_data, batch_size=2,shuffle=False)
    data_mean, data_std = get_mean_and_std(loader)
    print(data_mean,data_std)

    # import torchvision.transforms as T
    # train_data = NasaDataset(root_dir=root_data_dir,filelist="data_split/nadir_to_all/train.csv",mode="rad2rad")
    # print(len(train_data))
    # loader = DataLoader(train_data, batch_size=128,shuffle=False)
    # data_mean, data_std = get_mean_and_std(loader)
    # print(data_mean,data_std)
    # transform_func = T.Compose([
    # T.Normalize(mean=data_mean, std=data_std)
    # ])
    # # loader.dataset.set_transform(transform_func)
    # # loader.dataset.set_dis_code(True)
    # del loader
    # del train_data
    # train_data = NasaDataset(root_dir=root_data_dir,
    #             filelist="data_split/nadir_to_all/train.csv",
    #             mode="rad2rad",transform=transform_func)
    # print(len(train_data))
    # loader = DataLoader(train_data, batch_size=128,shuffle=False)
    # data_mean, data_std = get_mean_and_std(loader)
    # print(data_mean,data_std)    