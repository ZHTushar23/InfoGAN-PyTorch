import numpy as np
import csv


def create_csv_for_data(profile_dict=None,mode='cot',anlge_dict=None,output_file_dir=None):

    # extract the angles
    SZA1 = anlge_dict['SZA1']
    VZA1 = anlge_dict['VZA1']
    if mode=='rad2rad' or mode=='rad2rad2':
        SZA2 = anlge_dict['SZA2']
        VZA2 = anlge_dict['VZA2']

    for key in profile_dict.keys():
        profile_list = profile_dict[key]
        if mode=='rad2rad':
            # Initialize a list to hold the new rows
            new_rows = [['reflectance', 'reflectance2']]
        elif mode=='rad2rad2':
            # Initialize a list to hold the new rows
            new_rows = [['reflectance', 'reflectance2','p_num','sza','vza']]  
        elif mode=='rad2rad3':
            # Initialize a list to hold the new rows
            new_rows = [['reflectance', 'reflectance2','p_num','sza','vza']]        
        elif mode=='cot':
            new_rows=[['reflectance','cot']]

        for p_num in profile_list:
            for sza_index in range(len(SZA1)):
                for vza_index in range(len(VZA1)):

                    input_sza = SZA1[sza_index]
                    input_vza = VZA1[vza_index]
                    data_file_dir1 = "SZA_"+str(np.int16(input_sza))+"/"+"VZA_"+str(np.int16(input_vza))+"/"

                    if mode=='rad2rad':
                        # Get the corresponding output SZA2 and VZA2
                        output_sza = SZA2[sza_index]
                        output_vza = VZA2[vza_index]
                        data_file_dir2 = "SZA_"+str(np.int16(output_sza))+"/"+"VZA_"+str(np.int16(output_vza))+"/"
                    
                        for r in range(81):
                            rdata_filename1 = data_file_dir1+"100m_Profile_%05d_Patch_%05d_reflectance.npy"%(p_num,r)
                            rdata_filename2 = data_file_dir2+"100m_Profile_%05d_Patch_%05d_reflectance.npy"%(p_num,r)
                            new_row = [rdata_filename1,rdata_filename2]
                            new_rows.append(new_row)
                    elif mode=='rad2rad2':
                        # Get the corresponding output SZA2 and VZA2
                        output_sza = SZA2[sza_index]
                        output_vza = VZA2[vza_index]
                        data_file_dir2 = "SZA_"+str(np.int16(output_sza))+"/"+"VZA_"+str(np.int16(output_vza))+"/"
                    
                        for r in range(81):
                            rdata_filename1 = data_file_dir1+"100m_Profile_%05d_Patch_%05d_reflectance.npy"%(p_num,r)
                            rdata_filename2 = data_file_dir2+"100m_Profile_%05d_Patch_%05d_reflectance.npy"%(p_num,r)
                            new_row = [rdata_filename1,rdata_filename2,p_num,output_sza, output_vza ]
                            new_rows.append(new_row)
                    else:
                        for r in range(81):
                            rdata_filename1 = data_file_dir1+"100m_Profile_%05d_Patch_%05d_reflectance.npy"%(p_num,r)
                            cdata_filename  = data_file_dir1+"100m_Profile_%05d_Patch_%05d_cot.npy"%(p_num,r)
                            new_row = [rdata_filename1,cdata_filename]
                            new_rows.append(new_row)


        output_file_path = output_file_dir+'/%s.csv'%(key)

        # Write the new rows to a new CSV file
        with open(output_file_path, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(new_rows)

        print(f"Filtered and transformed data has been saved to {output_file_path}")


if __name__=="__main__":
    f = np.arange(1,31)
    g = np.arange(61,103)
    h = np.concatenate((f,g))
    profile_dict = {
        'train':h,
        'val': np.arange(51,61),
        'test': np.arange(31,51)
    }

    # specify angles for input output
    SZA1 = [4.0,4.0,4.0]
    SZA2 = [20.0,40.0,60.0]
    VZA1 = [0,0,0,0,0,0]
    VZA2 = [-15,15,30,-30,60,-60]

    anlge_dict = { 'SZA1':SZA1,
                    'VZA1':VZA1,
                    'SZA2':SZA2,
                    'VZA2':VZA2}
    mode='rad2rad2'
    output_file_dir= "data_split/nadir_to_all"

    create_csv_for_data(profile_dict=profile_dict,mode=mode,anlge_dict=anlge_dict,output_file_dir=output_file_dir)
