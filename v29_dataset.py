
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
        self.cls_embed,_ = get_embedder(32)
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

    def get_embedding_using_zenith_angle(self,sza,vza): 
        ee = (self.cls_embed(torch.tensor([sza]))+1)/2
        ff = (self.cls_embed(torch.tensor([vza]))+1)/2
        ee = ee.unsqueeze(1)
        ff = ff.unsqueeze(0)
        return ee*ff


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
        else:
            output_img = np.load(os.path.join(self.root_dir,self.filelist[idx][1]))

        if self.mode=='infoGAN':
            sza = np.float32(self.filelist[idx][3]) # SZA value
            vza = np.float32(self.filelist[idx][4]) # VZA value
            dis_c = self.get_embedding_using_zenith_angle(sza,vza)
            dis_c = dis_c.unsqueeze(0)

            #find index
            s = sza_full_list.index(sza)
            v = vza_full_list.index(vza)
            # convert to tensor
            # dis_c =  self.transform1(dis_c)
            
            # discrete labels
            idxx = np.zeros((11)) # ['num_dis_c'] = 11, Batch size
            # idxx = torch.zeros((11)) # ['num_dis_c'] = 11, Batch size
            idxx[s]=1
            idxx[v+4]=1

        # Convert to tensor
        input_img =  self.transform1(input_img)
        if self.mode=="cot":
            output_img = np.log(output_img+1)
        
        if self.mode!="single":            
            output_img =  self.transform1(output_img)

        if self.transform:
            input_img =  self.transform(input_img)      
        if self.mode == "infoGAN":
            input_img = torch.cat((input_img, dis_c), dim=0)
            sample = {'input':input_img,'target':output_img,'idxx':idxx}
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


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : False,
                'input_dims' :1,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim



if __name__=="__main__":

    root_data_dir ="/home/ztushar1/psanjay_user/multi-view-cot-retrieval/LES_MultiView_100m_64/"
    for fold in range(1):    
        # fold=0
        print("checking fold: ", fold)
        train_data = NasaDataset(root_dir=root_data_dir,filelist="data_split/nadir_to_all/val.csv",mode="infoGAN")
        print(len(train_data))
        loader = DataLoader(train_data, batch_size=2,shuffle=False)
        
        temp= []
        temp1= []   
        for i in range(len(loader.dataset)):
            data = loader.dataset[i]
            # get the data
            X, Z= data['input'], data['target']
            print(X.shape, Z.shape)
            break
        for i, data in enumerate(loader):
            # get the data
            X, Z= data['input'], data['target']
            print(X.shape, Z.shape)
            break
    

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