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

class NasaDataset(Dataset):
    """  Dataset types:
        1. 'cv_dataset'
        """

    def __init__(self, root_dir=None,profilelist=None, vza_list1=None, vza_list2=None, 
                sza_list1=None, sza_list2=None, patch_size=20,stride=2,transform=None,
                add_dis = None):
        self.root_dir    = root_dir
        self.profilelist = profilelist
        self.vza_list1   = vza_list1
        self.vza_list2   = vza_list2
        self.sza_list1   = sza_list1
        self.sza_list2   = sza_list2
        self.patch_size  = patch_size
        self.stride      = stride
        self.transform1  = T.Compose([T.ToTensor()])
        self.transform   = transform
        self.add_dis     = add_dis

    def __len__(self):
        return len(self.profilelist)

    def set_transform(self,t_func):
        self.transform   = t_func
        print("Normalization constants are applied.")
    
    def set_dis_code(self,add_dis):
        self.add_dis     = add_dis
        print("Discrete code added.")


    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        p_num = self.profilelist[idx]

        fname = self.root_dir+"/LES_profile_%05d.hdf5"%(p_num)
        hf = h5py.File(fname, 'r')
        # print(hf.keys()) 

        
        # cmask =np.empty((144,144),dtype=float) 
        temp  = np.nan_to_num(np.array(hf.get("Reflectance_100m_resolution")))
        temp2 = np.nan_to_num(np.array(hf.get("Native_Cloud_fraction_(100m)")))
        
        r_total1 = []
        r_total2 = []
        idxx_total=[]
        dis_c_total=[]
       
        for sza in self.sza_list1:
            s = sza_full_list.index(sza)

            for vza in self.vza_list1:
                v = vza_full_list.index(vza)
                r_data   = np.zeros((144,144,2), dtype=float) #2+4+7=13

                # reflectance at 0.66 um
                r_data[:,:,0]   = temp[s,v,0,:,:]
                # reflectance at 2.13 um
                r_data[:,:,1]   = temp[s,v,1,:,:]
                r_total1.append(r_data)

        for sza in self.sza_list2:
            s = sza_full_list.index(sza)
            for vza in self.vza_list2:
                v = vza_full_list.index(vza)

                r_data   = np.empty((144,144,2), dtype=float) 
                dis_c    = np.zeros((144,144,11), dtype=float) # 4+7=11 sza+vza

                # reflectance at 0.66 um
                r_data[:,:,0]   = temp[s,v,0,:,:]
                # reflectance at 2.13 um
                r_data[:,:,1]   = temp[s,v,1,:,:]
                r_total2.append(r_data)

                # assign 1 to sza and vza channels
                dis_c[:,:,s] = 1
                dis_c[:,:,v+4] = 1
                dis_c_total.append(dis_c)

                # discrete labels
                idxx = np.zeros((11)) # ['num_dis_c'] = 11, Batch size
                # idxx = torch.zeros((11)) # ['num_dis_c'] = 11, Batch size
                idxx[s]=1
                idxx[v+4]=1
                idxx_total.append(idxx)

        hf.close()

        # Extract patches
        # patches_r1      = [self.extract_patches(img) for img in r_total1]
        # patches_r2      = [self.extract_patches(img) for img in r_total2]

        patches_r1 =[]
        patches_r2 =[]
        patches_idxx = []
        patches_dis  = []
        
        for ccc1 in range(len(r_total1)):
            img                 = r_total1[ccc1]
            img2                = r_total2[ccc1]
            dd                  = dis_c_total[ccc1]
            extracted_patches  = self.extract_patches(img)
            patches_r1.extend(extracted_patches)
            patches_idxx+=[idxx_total[ccc1]]*len(extracted_patches)

             
            patches_r2.extend(self.extract_patches(img2))
            patches_dis.extend(self.extract_patches(dd))
        
        del extracted_patches
        del r_total1
        del r_total2
        del dis_c_total


        # Convert to tensor
        if self.transform1:
            patches_r1 = [self.transform1(patch) for patch in patches_r1]   
            patches_r2 = [self.transform1(patch) for patch in patches_r2] 
            patches_dis = [self.transform1(patch) for patch in patches_dis]
        if self.transform:
            patches_r1 = [self.transform(patch) for patch in patches_r1] 
        if self.add_dis:
            concatenated_patches = [torch.cat((patch1, patch2), dim=0) for patch1, patch2 in zip(patches_r1, patches_dis)]
            patches_r1 = concatenated_patches
            del concatenated_patches


        sample = {'rad_patches':patches_r1,'rad_patches2':patches_r2, 
                    'p_num':p_num, 'idxx':patches_idxx,}
        return sample


    def extract_patches(self, image):
        h, w, c = image.shape
        patches = []

        for i in range(0, h - self.patch_size + 1, self.stride):
            for j in range(0, w - self.patch_size + 1, self.stride):
                patch = image[i:i + self.patch_size, j:j + self.patch_size]
                patches.append(patch)

        return patches
    

if __name__=="__main__":
    f = np.arange(1,31)
    g = np.arange(61,103)
    h = np.concatenate((f,g))
    # print(type(h), h, h.shape)

    for fold in range(1):    
        # fold=0
        print("checking fold: ", fold)
        dataset_dir1 = "/nfs/rs/psanjay/users/ztushar1/LES_vers1_multiangle_results"
        # dataset_dir1 = "/home/local/AD/ztushar1/Data/LES_vers1_multiangle_results"
        # sza_list = [60.0,40.0,20.0,4.0]
        # vza_list1 = [0,0]
        # vza_list2 = [15,30]
        # sza_list1 = [4.0,4.0]
        # sza_list2 = [20.0, 40.0]
        vza_list1 = [0,0,0,0,0,0]
        vza_list2 = [15,30,60,-15,-30,-60]
        sza_list1 = [4.0,4.0, 4.0]
        sza_list2 = [20.0, 40.0, 60.0]
        profilelist = h
        train_data = NasaDataset(profilelist=profilelist,root_dir=dataset_dir1,
                        vza_list1 = vza_list1,vza_list2 = vza_list2, sza_list1 = sza_list1,sza_list2 = sza_list2,
                                patch_size=64,stride=10)
        print(len(train_data))
        loader = DataLoader(train_data, batch_size=2,shuffle=False)
        
        temp= []
        temp1= []   
        for i in range(len(loader.dataset)):
            data = loader.dataset[i]
            # get the data
            X, Z, idxx = data['rad_patches'],data['rad_patches2'], data['idxx']
            print(len(X),len(Z), len(idxx))

            print(X[0].shape)
            print(Z[0].shape)
            print(idxx[0].shape)

            print(idxx[0])
            print(idxx[82])
            print(idxx[81*2])
            print(idxx[81*3])

            print(torch.max(X[0]), torch.min(X[0]))
            # target = idxx[0].to(dtype=torch.float32)

            break

        from v2_utils import get_mean_and_std
        data_mean, data_std = get_mean_and_std(loader)
        transform_func = T.Compose([
        T.Normalize(mean=data_mean, std=data_std)
        ])
        # loader.dataset.set_transform(transform_func)
        # loader.dataset.set_dis_code(True)

        # data_mean, data_std = get_mean_and_std(loader)
        # print(data_mean,data_std)
        del loader
        del train_data
        train_data = NasaDataset(profilelist=profilelist,root_dir=dataset_dir1,
                vza_list1 = vza_list1,vza_list2 = vza_list2, sza_list1 = sza_list1,sza_list2 = sza_list2,
                        patch_size=64,stride=10,transform=transform_func,add_dis=True)
        loader = DataLoader(train_data, batch_size=8,shuffle=False)

        # for i in range(len(loader.dataset)):
        #     data = loader.dataset[i]
        #     # get the data
        #     X, Z, idxx = data['rad_patches'],data['rad_patches2'], data['idxx']
        #     print(len(X),len(Z), len(idxx))

        #     print(X[0].shape)
        #     print(Z[0].shape)
        #     print(idxx[0].shape)

        #     print(idxx[0])
        #     print(idxx[82])
        #     print(idxx[81*2])
        #     print(idxx[81*3])

        #     print(torch.max(X[0]), torch.min(X[0]))

        #     break
    #     # break  
    #     el = [torch.max(patch[0,:,:]).item() for patch in X]  
    #     em = [torch.min(patch[0,:,:]).item() for patch in X] 
    #     temp.append(np.max(el))
    #     temp1.append(np.min(em))
    # print(np.max(temp),np.min(temp1))
    
        import random
        for _, data in enumerate(loader, 1):
            # get the data
            X, Z, idxx = data['rad_patches'],data['rad_patches2'], data['idxx']

            # Shuffle the indices
            indices = list(range(len(X)))
            random.shuffle(indices)
            for cc in range(0, len(indices), 8):
                chunk_indices = indices[cc:cc+8]
                chunk_tensors = [X[ix] for ix in chunk_indices]
                concatenated_tensor = torch.cat(chunk_tensors, dim=0)  # Concatenating along the batch dimension
                print(concatenated_tensor.shape)

                chunk_tensors = [Z[ix] for ix in chunk_indices]
                concatenated_tensor = torch.cat(chunk_tensors, dim=0)  # Concatenating along the batch dimension
                print(concatenated_tensor.shape)

                chunk_tensors = [idxx[ix] for ix in chunk_indices]
                concatenated_tensor = torch.cat(chunk_tensors, dim=0)  # Concatenating along the batch dimension
                print(concatenated_tensor.shape)
                break
            break
        # for _, data in enumerate(loader, 1):
        #     # for i in range(len(train_loader.dataset)):
        #     #     data = train_loader.dataset[i]
        #     r_train, m_train = data['rad_patches'],data['rad_patches2']
        #     print(len(r_train))
        #     break
    
    # dataset_dir2 = "/home/local/AD/ztushar1/Data/ncer_fill3"

    # test_data = NasaDataset(root_dir=dataset_dir2)
    # loader = DataLoader(test_data, batch_size=10,shuffle=False)
    # print(len(loader.dataset))
    # for i in range(len(loader.dataset)):
    #     data = loader.dataset[i]
    #     # get the data
    #     X, Y, Z = data['reflectance'],data['cmask'],data['patches']
    #     print(Y.shape, X.shape,Z[0].shape)
    #     break  
    
    # temp1 = []
    # temp2 = []

    # temp3, temp4 = [],[]
    # for i in range(len(loader.dataset)):
    #     data = loader.dataset[i]
    #     # get the data
    #     Y1= data['reflectance']
        
    #     # print(torch.max(Y))
        
    #     for k in range(len(Y1)):
    #         if i==0 and k==0:
    #             temp1=torch.from_numpy(Y1[k][1])
    #         else:
    #             temp1 = torch.cat((temp1,torch.from_numpy(Y1[k][1])))
    #         # temp3 = torch.cat((temp3,Y2[:]))

    # print("COT Quantile: ")
    # print(torch.quantile(temp1[:],0.9))