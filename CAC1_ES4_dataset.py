
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

def get_5bit_code(sza,vza):
    '''
    takes sza and vza values, finds their index and computes 5bit code

    '''
    sza_full_list = [60.0,40.0,20.0,4.0]
    vza_full_list = [60,30,15,0,-15,-30,-60]
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
    code = torch.tensor(dis_c_list, dtype=torch.float32)

    return code

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
        self.samples, self.samples2, self.targets = self._make_dataset()

    def _make_dataset(self):
        fnames, fnames2, labels, labels2, target = [], [], [], [],[]

        for idx in range(len(self.filelist)):
            target_fnames = os.path.join(self.root_dir,self.filelist[idx][1]) #cot
            cls_fnames    = os.path.join(self.root_dir,self.filelist[idx][0]) #radiance
           
            sza = int(float(self.filelist[idx][3]))
            vza = int(self.filelist[idx][4])

            sza2 = sza
            
            for vza2 in vza_full_list:

                if vza2==vza:
                    continue
                else:
                    target += [target_fnames]
                    fnames += [cls_fnames]
                    labels += [get_5bit_code(sza,vza)] # assigning class labels based on domain. first domain has label 0, next is 1.
                    nameeee = "SZA_"+str(np.int16(sza2))+"/"+"VZA_"+str(vza2)+"/"

                    cls_fnames2 = os.path.join(self.root_dir,nameeee+self.filelist[idx][0][-46:])
                    fnames2 += [cls_fnames2]
                    labels2 +=[get_5bit_code(sza2,vza2)]

        return list(zip(fnames, labels)), list(zip(fnames2, labels2)), list(target)


    def __len__(self):
        return len(self.targets)
        # return len(self.filelist)

    def __getitem__(self, index):
        fname, label = self.samples[index]
        fname2, label2 = self.samples2[index]
        fname_cot = self.targets[index]
        img   = np.load(fname)
        img2  = np.load(fname2)

        output_img = np.load(fname_cot)
        output_img = np.log(output_img[:,:,0]+1)

        img = self.transform1(img)
        img2 = self.transform1(img2)

        output_img = self.transform1(output_img)

        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)

        sample = {'input1':img, 'code1':label,
                'input2':img2, 'code2':label2,
                'target':output_img}
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
    data_split_dir = "/home/ztushar1/psanjay_user/multi-view-cot-retrieval/"
    data_split_sub_dir = "data_split/cot_sza_20_vza_all/"

    data_mean, data_std = torch.tensor([0.1410, 0.1020], dtype=torch.float64), torch.tensor([0.1596, 0.0854], dtype=torch.float64)
    transform_func = T.Compose([T.Normalize(mean=data_mean, std=data_std)])

    train_data = NasaDataset(root_dir=root_data_dir,
                filelist=data_split_dir+data_split_sub_dir+"/train.csv",
                mode="infoGAN_cot",transform=transform_func)
    for fold in range(1):    
        # fold=0
        print("checking fold: ", fold)
        print(len(train_data))
        loader = DataLoader(train_data, batch_size=2,shuffle=False)
        
        temp= []
        temp1= []   
        for i in range(len(loader.dataset)):
            data = loader.dataset[i]
            # get the data
            X, Z, codee = data['input1'], data['target'], data['code1']
            print(X.shape, Z.shape, codee.shape)
            break
        for i, data in enumerate(loader):
            # get the data
            X, Z, CC = data['input1'], data['target'], data['code1']
            print(X.shape, Z.shape, CC.shape)
            break
        for i in range(len(loader.dataset)):
            data = loader.dataset[i]
            # get the data
            X, Z, codee = data['input2'], data['target'], data['code2']
            print(X.shape, Z.shape, codee.shape)
            break