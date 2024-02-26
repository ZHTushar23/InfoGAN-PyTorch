'''
    Author: Zahid Hassan Tushar
    email: ztushar1@umbc.edu
'''
import os
import csv
from torch.utils.data import Dataset, random_split, DataLoader
import torch
import numpy as np
import torchvision.transforms as T
def _read_csv(path: str):
    """Reads a csv file, and returns the content inside a list of dictionaries.
    Args:
      path: The path to the csv file.
    Returns:
      A list of dictionaries. Each row in the csv file will be a list entry. The
      dictionary is keyed by the column names.
    """
    with open(path, "r") as f:
        return list(csv.DictReader(f))

class NasaDataset(Dataset):
    def __init__(self, root_dir,mode="train",transform_cot=None):
        self.root_dir = root_dir
        self.csv_file = _read_csv(root_dir+mode+".csv")
        # self.filelist = os.listdir(root_dir)
        self.transform1 = T.Compose([T.ToTensor()])
        if transform_cot:
            self.transform2=transform_cot
    
    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        reflectance_name = os.path.join(self.root_dir,self.csv_file[idx]['reflectance'])
        reflectance      = np.load(reflectance_name)[:,:,:1]
        # print(reflectance.shape)

        # cot profile
        cot_name         = os.path.join(self.root_dir,self.csv_file[idx]['cot'])
        cot_data         = np.load(cot_name)[:,:,:1]

        p_num = torch.tensor(int(self.csv_file[idx]['Profile']))

        # Angle Information
        sza = self.csv_file[idx]['SZA']
        vza = self.csv_file[idx]['VZA']

        sza_temp = torch.tensor([float(sza)])
        vza_temp = torch.tensor([float(vza)])

        # Convert to tensor
        reflectance = self.transform1(reflectance)
        cot_data    = self.transform1(cot_data)

        sample = {'reflectance': reflectance, 'cot': cot_data,"sza":sza_temp,
                "vza":vza_temp,"p_num":p_num, 
                "name":self.csv_file[idx]['cot'].split('/')[2].split('.')[0] }
        return sample

if __name__=="__main__":
    # dataset_dir = "/nfs/rs/psanjay/users/ztushar1/COT_CER_Joint_Retrievals/one_thousand_profiles/Refl"
    dataset_dir = "/nfs/rs/psanjay/users/ztushar1/multi-view-cot-retrieval/LES102_MultiView_100m_F2/"
    dataset_dir ="/home/local/AD/ztushar1/LES102_MultiView_100m_F2/"
    train_data = NasaDataset(root_dir=dataset_dir,mode="val")
    loader = DataLoader(train_data, batch_size=10,shuffle=False)
    print(len(loader.dataset))
    sample=loader.dataset[0]
    print(sample['reflectance'].shape)