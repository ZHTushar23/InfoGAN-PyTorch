
'''
    Author: Zahid Hassan Tushar
    email: ztushar1@umbc.edu
'''


# import libraries
from v13_utilities import *
from v18_dataset import *
from visualization import *
import os
from models.cloud_model18 import Generator

# dataset dir
dataset_dir1 = "/nfs/rs/psanjay/users/ztushar1/LES_vers1_multiangle_results"
f = np.arange(1,31)
g = np.arange(61,103)
h = np.concatenate((f,g))

sza_full_list = [60.0,40.0,20.0,4.0]
vza_full_list = [60,30,15,0,-15,-30,-60]



def get_profile_pred(model,X_test,Y_test,transform2,patch_size=64,stride=10):


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
    map = np.zeros_like(np.squeeze(Y_test))
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



            patch_pred = get_predictions(model=model,X_test=patch,Y_test=None)
            # print("pred patch shape: ",patch_pred.shape)
            # print("pred patch min max: ",np.min(patch_pred[1,:,:]),np.max(patch_pred[1,:,:]))
            # map[row_start:row_end,col_start:col_end] =map[row_start:row_end,col_start:col_end] +np.ones((10,10))
          
            map[:,row_start:row_end,col_start:col_end] +=1
            Y_pred[:,row_start:row_end,col_start:col_end] +=patch_pred

    

    # Y_pred = Y_pred[:,2:-2,2:-2]/map[:,2:-2,2:-2]
    # Y_pred = Y_pred[:,2:-2,2:-2]
    Y_pred = Y_pred/map
    # Y_pred[2,:,:] = (Y_pred[2,:,:]>0.5)*1

    Y_test = Y_test.cpu().detach().numpy()

    return Y_test, Y_pred




# define a function that takes model's name and do the rests
def run_test(params):

    # create directory to save checkpoints
    saved_model_root_dir = "v17_saved_model/sza_"+str(params['sza_list2'])+"_vza_"+str(params['vza_list2'])


    # sza_values_str = '_'.join(map(str, args.sza))
    # vza_values_str = '_'.join(map(str, args.vza))

    # Check if the GPU is available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    print(f'Main Selected device: {device}')

    # compute normalization constant based on the training set

    vza_list1 = params['vza_list1']
    vza_list2 = params['vza_list2']
    sza_list1 = params['sza_list1']
    sza_list2 = params['sza_list2']
    profilelist = h


    # compute std and mean for the new train split
    train_data = NasaDataset(profilelist=h,root_dir=dataset_dir1,
                    vza_list1 = vza_list1,vza_list2 = vza_list2, sza_list1 = sza_list1,sza_list2 = sza_list2,
                            patch_size=64,stride=10)

    loader = DataLoader(train_data, batch_size=params['batch_size'],shuffle=True)
    data_mean, data_std = get_mean_and_std(loader)

    transform_func = T.Compose([
    T.Normalize(mean=data_mean, std=data_std)
    ])

    del loader, train_data

    # Create the generator network.
    netG = Generator(13).to(device)

    total_loss=[]
    for fold in range(5):
        saved_model_dir = saved_model_root_dir+"/rfold_%01d"%(fold)
        load_path = saved_model_dir+'/model_epoch_%d_{}'.format(params['dataset']) %(475)
        # load_path  = 'checkpoint/model_final_{}'.format(params['dataset'])
        # Load the checkpoint file
        state_dict = torch.load(load_path)
        # Load the trained generator weights.
        netG.load_state_dict(state_dict['netG'])
        # print(netG)
        # Generate image.
        netG.eval()

        dir_name = saved_model_dir+"/full_profile_fold_"+str(fold)
        try:
            os.makedirs(dir_name)
        except FileExistsError:
            print("folder already exists")


        # load profile
        p_num = 35
        fname = dataset_dir1+"/LES_profile_%05d.hdf5"%(p_num)
        hf = h5py.File(fname, 'r')
        # print(hf.keys()) 

        
        # cmask =np.empty((144,144),dtype=float) 
        temp  = np.nan_to_num(np.array(hf.get("Reflectance_100m_resolution")))
        temp2 = np.nan_to_num(np.array(hf.get("Native_Cloud_fraction_(100m)")))
        hf.close()

        for sza in sza_list1:
            s = sza_full_list.index(sza)
            for vza in vza_list1:
                v = vza_full_list.index(vza)

                r_data_ip   = np.empty((144,144,2), dtype=float) 

                # reflectance at 0.66 um
                r_data_ip[:,:,0]   = temp[s,v,0,:,:]
                # reflectance at 2.13 um
                r_data_ip[:,:,1]   = temp[s,v,1,:,:]
        
        for sza in sza_list2:
            s = sza_full_list.index(sza)
            for vza in vza_list2:
                v = vza_full_list.index(vza)

                r_data_op   = np.empty((144,144,2), dtype=float) 

                # reflectance at 0.66 um
                r_data_op[:,:,0]   = temp[s,v,0,:,:]
                # reflectance at 2.13 um
                r_data_op[:,:,1]   = temp[s,v,1,:,:]

        for aa in range (len(sza_list2))
            profile, pred = get_profile_pred(model,r_data_ip[aa],r_data_op[aa],transform_func)


            limit1 = [0,2]
            limit2 = [0,1]
            use_log=False

            # Plot Input Radiance
            fname = dir_name+"/rad066_profile_%03d.png"%(p_num )
            plot_cot2(cot=r_data_ip[:,:,0],title="Radiance at 0.66um",fname=fname,use_log=use_log,limit=limit1)

            fname = dir_name+"/op_rad066_profile_%03d.png"%(p_num )
            plot_cot2(cot=profile[0,:,:],title="Radiance at 0.66um",fname=fname,use_log=use_log,limit=limit1)

            fname = dir_name+"/op_pred_rad066_profile_%03d.png"%(p_num )
            plot_cot2(cot=pred[0,:,:],title="Radiance at 0.66um",fname=fname,use_log=use_log,limit=limit1)

            fname = dir_name+"/abs_error_066_profile_%03d.png"%(p_num )
            plot_cot2(cot=(profile[0,:,:]-pred[0,:,:]),title="Absolute Error 0.66 um",fname=fname,use_log=use_log,limit=limit2)    

            # Plot Input Radiance
            fname = dir_name+"/rad213_profile_%03d.png"%(p_num )
            plot_cot2(cot=r_data_ip[:,:,1],title="Radiance at 2.13um",fname=fname,use_log=use_log,limit=limit1)

            fname = dir_name+"/op_rad213_profile_%03d.png"%(p_num )
            plot_cot2(cot=profile[1,:,:],title="Radiance at 2.13um",fname=fname,use_log=use_log,limit=limit1)

            fname = dir_name+"/op_pred_rad213_profile_%03d.png"%(p_num )
            plot_cot2(cot=pred[1,:,:],title="Radiance at 2.13um",fname=fname,use_log=use_log,limit=limit1)

            fname = dir_name+"/abs_error_213_profile_%03d.png"%(p_num )
            plot_cot2(cot=np.abs(profile[1,:,:]-pred[1,:,:]),title="Absolute Error 2.13 um",fname=fname,use_log=use_log,limit=limit2)          

    print("Done!")



if __name__=="__main__":
    model_name = "cam"
    # model_filenames= ["cam_fold_0_20240319_235931.pth",
    #                 "cam_fold_1_20240320_004916.pth",
    #                 "cam_fold_2_20240320_020050.pth",
    #                 "cam_fold_3_20240320_025421.pth",
    #                 "cam_fold_4_20240320_034852.pth"]
    # model_filenames= [
    #     # "cam_fold_0_20240321_170151.pth",
    #                 "cam_fold_1_20240321_174205.pth",
    #                 "cam_fold_2_20240321_182402.pth",
    #                 "cam_fold_3_20240321_190157.pth",
    #                 "cam_fold_4_20240321_195702.pth"]  #L1
    model_filenames= ["cam_fold_0_20240401_141821.pth",
                    "cam_fold_1_20240401_153231.pth",
                    "cam_fold_2_20240401_170157.pth",
                    "cam_fold_3_20240401_180901.pth",
                    "cam_fold_4_20240401_192955.pth"]     # ssim w penalty 6
    

    f=4
    sza_list  = [4.0]
    vza_list1 = [0] # [30,0,-30] 
    vza_list2 = [15]


    # model_filenames= ["cam_fold_0_20240408_010912.pth",
    #                 "cam_fold_1_20240408_020145.pth",
    #                 "cam_fold_2_20240408_025128.pth",
    #                 "cam_fold_3_20240408_040155.pth",
    #                 "cam_fold_4_20240408_044517.pth"]     # ssim w penalty 6
    

    # f=1
    # sza_list  = [4.0]
    # vza_list1 = [0] # [30,0,-30] 
    # vza_list2 = [60]

    # model_filenames= ["cam_fold_0_20240319_235949.pth",
    #                 "cam_fold_1_20240320_010552.pth",
    #                 "cam_fold_2_20240320_014611.pth",
    #                 "cam_fold_3_20240320_022530.pth",
    #                 "cam_fold_4_20240320_034454.pth"]

    # model_filenames= [
    #     # "cam_fold_0_20240321_170151.pth",
    #                 "cam_fold_1_20240321_182146.pth",
    #                 "cam_fold_2_20240321_191429.pth",
    #                 "cam_fold_3_20240321_194553.pth",
    #                 "cam_fold_4_20240321_202556.pth"]  # L1

    # a= [0]
    # b = [-15]
    # f=2

    # model_filenames= ["cam_fold_0_20240320_000032.pth",
    #                 "cam_fold_1_20240320_011438.pth",
    #                 "cam_fold_2_20240320_015857.pth",
    #                 "cam_fold_3_20240320_023302.pth",
    #                 "cam_fold_4_20240320_032307.pth"]

    # model_filenames= [
    #     # "cam_fold_0_20240320_000032.pth",
    #                 "cam_fold_1_20240321_182626.pth",
    #                 "cam_fold_2_20240321_191028.pth",
    #                 "cam_fold_3_20240321_201229.pth",
    #                 "cam_fold_4_20240321_204651.pth"]  #L1
    # a= [0]
    # b = [30]
    # f=3

    # model_filenames= ["cam_fold_0_20240320_000109.pth",
    #                 "cam_fold_1_20240320_014435.pth",
    #                 "cam_fold_2_20240320_032235.pth",
    #                 "cam_fold_3_20240320_040106.pth",
    #                 "cam_fold_4_20240320_050325.pth"]

    # model_filenames= [
    #     # "cam_fold_0_20240320_000109.pth",
    #                 "cam_fold_1_20240321_180946.pth",
    #                 "cam_fold_2_20240321_185432.pth",
    #                 "cam_fold_3_20240321_195223.pth",
    #                 "cam_fold_4_20240321_214544.pth"] # L1

    # a= [0]
    # b = [-30]
    # f=0

    out = {"model_filenames":model_filenames, "model_name":model_name,
            "vza1":vza_list1, "vza2":vza_list2,"sza":sza_list,
            "f":f,"fig":"20240408_010912"}
    run_test(out)
    # for p_b in range(0, 36, 2):
    #     print(p_b)
