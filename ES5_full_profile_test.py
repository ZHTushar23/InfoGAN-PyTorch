
'''
    Author: Zahid Hassan Tushar
    email: ztushar1@umbc.edu
'''


# import libraries
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import os
import h5py
import csv
from ES5_config import params
from models.es5_model import Generator
import numpy as np
# dataset dir
dataset_dir1 = "/nfs/rs/psanjay/users/ztushar1/LES_vers1_multiangle_results"
sza_full_list = [60.0,40.0,20.0,4.0]
vza_full_list = [60,30,15,0,-15,-30,-60]


def get_profile_pred(model,X_test,Y_test,style_code,transform2,device,patch_size=64,stride=10):
    style_code = style_code.to(device,dtype=torch.float32)
    style_code = torch.unsqueeze(style_code,0)
    # stride = 2
    img_width = X_test.shape[1]
    img_height = X_test.shape[0]
    patch_height,patch_width = patch_size,patch_size

    r = np.int32(np.ceil((img_height-patch_height)/stride))+1
    c = np.int32(np.ceil((img_width-patch_width)/stride))+1

    #2 convert to tensor
    
    X_test = TF.to_tensor(X_test)
    # print(torch.min(X_test))
    # print(torch.max(X_test))
    Y_test = TF.to_tensor(Y_test)
    map = np.zeros_like(Y_test)
    Y_pred = np.zeros_like(map)
    #3 Normalize data
    X_test  = transform2(X_test)

    # patch_holder = np.empty((r*c,patch_height,patch_width),dtype=float) 
    for row in range(r):
        for col in range(c):
            row_start = min(row*stride,img_height-patch_height)
            row_end = row_start+patch_height
            col_start =  min(col*stride,img_width-patch_width)
            col_end = col_start+patch_width
            patch = X_test[:,row_start:row_end,col_start:col_end]

            # if model_name=="okamura":
            #     label = Y_test[0:2,row_start+2:row_end-2,col_start+2:col_end-2]
            # elif model_name=="okamura2":
            #     label = Y_test[0:2,row_start+1:row_end-1,col_start+1:col_end-1]
            # else:
            #     label = Y_test[0:2,row_start:row_end,col_start:col_end]


            patch   = patch.to(device,dtype=torch.float32)            
            patch = torch.unsqueeze(patch,0)

            with torch.no_grad():
                patch_pred = model(patch,style_code).detach().cpu().numpy()
            # print("pred patch shape: ",patch_pred.shape)
            # print("pred patch min max: ",np.min(patch_pred[1,:,:]),np.max(patch_pred[1,:,:]))
            # map[row_start:row_end,col_start:col_end] =map[row_start:row_end,col_start:col_end] +np.ones((10,10))
          
            map[:,row_start:row_end,col_start:col_end] +=1
            Y_pred[:,row_start:row_end,col_start:col_end] +=patch_pred[0]

    

    # Y_pred = Y_pred[:,2:-2,2:-2]/map[:,2:-2,2:-2]
    # Y_pred = Y_pred[:,2:-2,2:-2]
    Y_pred = Y_pred/map
    # Y_pred[2,:,:] = (Y_pred[2,:,:]>0.5)*1

    Y_test = Y_test.cpu().detach().numpy()
    mse_loss = np.average((Y_pred-Y_test)**2)

    scores = {"mse":mse_loss}

    return Y_test, Y_pred, scores




# define a function that takes model's name and do the rests
def run_test(out):

    # Check if the GPU is available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    print(f'Main Selected device: {device}')

    # compute normalization constant based on the training set

    vza_list1 = out["vza1"]  # [30,0,-30] 
    sza_list1 = out["sza1"]  # [30,0,-30] 

    # compute std and mean for the new train split
    data_mean, data_std = torch.tensor([0.1600, 0.1121], dtype=torch.float64), torch.tensor([0.2106, 0.1189], dtype=torch.float64)

    # define transform function with new std and mean
    transform_func = T.Compose([
    T.Normalize(mean=data_mean, std=data_std)
    ])

    # Create the generator network.
    netG = Generator(in_ch=2,out_ch=1).to(device)

    # load model
    saved_model_dir = 'es5_saved_model'
    dir_name = saved_model_dir+"/full_profile_"
    try:
        os.makedirs(dir_name)
    except FileExistsError:
        print("folder already exists")

    header = ["Name","MSE","STD"]
    losscsv = "/loss_log.csv"

    csv_name = dir_name+losscsv
    with open(csv_name, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)    
        writer.writerow(header) 

        total_loss=[]
        for fold in range(2):
            # if fold!=out["f"]:
            #     continue

            saved_model_name = "fold_%01d"%(fold)+'/model_final_{}'.format(params['dataset'])
            saved_model_path = os.path.join(saved_model_dir,saved_model_name)
            # Load the checkpoint file
            state_dict = torch.load(saved_model_path,map_location=torch.device('cpu'))
            # Load the trained generator weights.
            netG.load_state_dict(state_dict['netG'])
            fold_loss = []
            for p_num in range(31,51):
                # load profile
                fname = dataset_dir1+"/LES_profile_%05d.hdf5"%(p_num)
                hf = h5py.File(fname, 'r')
                # print(hf.keys()) 

                
                # cmask =np.empty((144,144),dtype=float) 
                temp  = np.nan_to_num(np.array(hf.get("Reflectance_100m_resolution")))
                temp2 = np.nan_to_num(np.array(hf.get("Cloud_optical_thickness_(100m resolution)")))
                hf.close()

                for sza in sza_list1:
                    s = sza_full_list.index(sza)
                    for vza in vza_list1:
                        v = vza_full_list.index(vza)

                        r_data_ip   = np.empty((144,144,2), dtype=float) 
                        cot         = np.log(temp2 +1)

                        sza_bits = format(s, '02b')
                        vza_bits = format(v, '03b')
                        # Concatenate the bits to form the 5-bit code
                        dis_c = sza_bits + vza_bits
                        # Convert dis_c to a list of integers (0 or 1)
                        dis_c_list = [int(bit) for bit in dis_c]
                        # Convert the list to a torch tensor
                        style_code = torch.tensor(dis_c_list, dtype=torch.float32)                         

                        # reflectance at 0.66 um
                        r_data_ip[:,:,0]   = temp[s,v,0,:,:]
                        # reflectance at 2.13 um
                        r_data_ip[:,:,1]   = temp[s,v,1,:,:]

                        profile, pred, scores = get_profile_pred(netG,r_data_ip,cot,style_code,transform_func,device)
                        fold_loss.append(scores['mse'])

            print("Fold: ",fold, " Mean test Loss: ", np.average(fold_loss), " Std: ", np.std(fold_loss))
            writer.writerow([saved_model_name,np.average(fold_loss),np.std(fold_loss)])
            total_loss.append(np.average(fold_loss))
        print(" Mean test Loss: ", np.average(total_loss), " Std: ", np.std(total_loss))
        writer.writerow([saved_model_name,np.average(total_loss), np.std(total_loss)])
    print("Done!")

    #     else:
    #         test_loss = test_model2(model, test_loader,device,2)
    #     print("test Loss: ",test_loss)
    #     total_loss.append(test_loss)

    # print(" Mean test Loss: ", np.average(total_loss), " Std: ", np.std(total_loss))


if __name__=="__main__":

    for sza_list1 in sza_full_list:
        for vza_list1 in vza_full_list:
            # sza_list1  = [40.0]
            # vza_list1  = [0]

            out = {
                    "sza1":[sza_list1], "vza1":[vza_list1]}
            print("SZA: ", sza_list1, " VZA: ",vza_list1)
            run_test(out)
    # for p_b in range(0, 36, 2):
    #     print(p_b)
