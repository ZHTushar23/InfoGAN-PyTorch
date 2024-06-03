'''
    Author: Zahid Hassan Tushar
    email: ztushar1@umbc.edu
'''

'''
This script is used to generate visualization for the experiments. It will be updated accordingly.
At the moment, it generates five plots [input reflectance data, cot data, predictions, cloud mask]
'''
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as Lines
from matplotlib import colors
import matplotlib.ticker as plticker
# import seaborn as sns
np.random.seed(0)
sza = [60.0,40.0,20.0,4.0]
vza = [60,30,15,0,-15,-30,-60]



def plot_2d_color(GT,P,fname, bins=200, limits=[0,7], labels=["True COT","CAM COT"]):

    H, xedges, yedges = np.histogram2d(GT, P, bins=bins,density=True, range=[limits,limits])
    # Histogram does not follow Cartesian convention (see Notes),
    # therefore transpose H for visualization purposes.
    H = H.T

    fig, ax1 = plt.subplots(ncols=1, sharey=True)
    plt.imshow(H, interpolation='nearest', origin='lower',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.colorbar()
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.savefig(fname)
    plt.close()


def plot_2d_color2(GT,P,fname, bins=200, limits=[0,7], labels=["True COT","CAM COT"]):

    H, yedges, xedges = np.histogram2d(P, GT, bins=bins,density=True, range=[limits,limits])
    # Plot histogram using pcolormesh
    fig, ax1 = plt.subplots(ncols=1, sharey=True)
    im = ax1.pcolormesh(xedges, yedges, H, cmap='jet')
    # ax1.plot(x, 2*np.log(x), 'k-')
    ax1.set_xlim(limits[0],limits[1])
    ax1.set_ylim(limits[0],limits[1])
    ax1.set_xlabel(labels[0])
    ax1.set_ylabel(labels[1])
    # ax1.set_title('histogram2d')
    # ax1.grid()
    plt.colorbar(im)
    plt.savefig(fname)
    plt.close()




def plot_five( gt, predictions,fname,use_log,limit):
    '''
    args: 
    gt    = 5 x 2 x 10 x 10  [5 x C x H x W] (C 0 is COT, C 1 is cloud mask)
    predictions =  5 x 1 x 10 x 10 [5 x C x H x W] (numpy array)
    
    '''
    # check if input is numpy array or not, check for dimensions

    # plot
    eps = 0.01

    fig,axs = plt.subplots(3,5)
    # cmap = 'seismic'
    # cmap ='bwr'
    for row in range(3):
        for col in range(5):
            ax = axs[row,col]

            # Plot the data            
            if row == 0:
                Z = gt[col,0,:,:]
                if use_log:
                    pcm = ax.pcolor(Z, norm=colors.LogNorm(vmin=limit[0], vmax=limit[1]),
                    cmap='jet', shading='auto')
                else:
                    pcm = ax.pcolor(Z, cmap='jet', shading='auto')
                    pcm.set_clim(limit[0],limit[1]) # normalized range
                    # pcm.set_clim(0,360) # denormalized regular range
            
            elif row == 1:
                Z = predictions[col,0,:,:]
                if use_log:
                    pcm = ax.pcolor(Z, norm=colors.LogNorm(vmin=limit[0], vmax=limit[1]),
                    cmap='jet', shading='auto')
                else:
                    pcm = ax.pcolor(Z, cmap='jet', shading='auto')
                    pcm.set_clim(limit[0],limit[1]) # normalized range
                    # pcm.set_clim(-2.5,13) # normalized range
                    # pcm.set_clim(0,360) # denormalized regular range

            elif row == 2:
                Z = gt[col,1,:,:]
                pcm = ax.pcolor(Z, cmap='jet', shading='auto')
                pcm.set_clim(0,1)
                      
              
     
            # Place labels and set limits on the axis 
            if row== 0 and col==0:
                ax.set(ylabel='COT')
                ax.axes.yaxis.set_visible(True)
                ax.axes.xaxis.set_visible(False)

            elif row== 1 and col==0:
                ax.set(ylabel='Pred')
                ax.axes.yaxis.set_visible(True)
                ax.axes.xaxis.set_visible(False)

            elif  row == 2 and col==0:
                ax.set(ylabel='C Mask')
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(True)
            else:
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)


        fig.colorbar(pcm, ax=axs[row,:])

    # plt.show()
    # fname = "results.png"
    plt.savefig(fname)
    plt.close()


def plot_cot(cot,title,fname,use_log,limit):
    fig,axs = plt.subplots(1,1)
    Z = cot
    if use_log:
        pcm = axs.pcolor(Z, norm=colors.LogNorm(vmin=limit[0], vmax=limit[1]),cmap='jet', shading='auto')
    else:
        pcm = axs.pcolor(Z, cmap='jet', shading='auto')
        pcm.set_clim(limit[0],limit[1])
    # pcm.set_clim(-2.5,13) # normalized range
    # pcm.set_clim(0,360) # denormalized regular range
    axs.set(title=title)
    axs.axes.yaxis.set_visible(True)
    axs.axes.xaxis.set_visible(True)
    fig.colorbar(pcm, ax=axs)
    plt.savefig(fname)
    plt.close()

def plot_cot2(cot,title,fname,use_log,limit):
    #     # Calculate aspect ratio based on data shape
    # aspect_ratio = cot.shape[1] / cot.shape[0]

    # Set figure size based on desired width and aspect ratio
    # fig_width = 8  # Adjust this value as needed
    # fig_height = fig_width / aspect_ratio
    # fig, axs = plt.subplots(1, 1, figsize=(fig_width, fig_height))  # Set figsize here
    fig, axs = plt.subplots(1, 1)
   
    # fig,axs = plt.subplots(1,1)
    Z = cot
    if use_log:
        pcm = axs.pcolor(Z, norm=colors.LogNorm(vmin=limit[0], vmax=limit[1]),cmap='jet', shading='auto')
    else:
        pcm = axs.pcolor(Z, cmap='jet', shading='auto')
        pcm.set_clim(limit[0],limit[1])
    # axs.set_title(title, fontsize=17)
    axs.tick_params(axis='both', which='major', labelsize=17)
    axs.axes.yaxis.set_visible(True)
    axs.axes.xaxis.set_visible(True)
    # fig.colorbar(pcm, ax=axs)
    cbar = fig.colorbar(pcm, ax=axs)
    cbar.ax.tick_params(labelsize=17)
    # cbar.set_label( fontsize=17)
    plt.savefig(fname)
    plt.close()

def plot_cot3(cot, title, fname, use_log, limit):
        # Calculate aspect ratio based on data shape
    aspect_ratio = cot.shape[1] / cot.shape[0]

    # Set figure size based on desired width and aspect ratio
    fig_width = 8  # Adjust this value as needed
    fig_height = fig_width / aspect_ratio
    fig, axs = plt.subplots(1, 1, figsize=(fig_width, fig_height))  # Set figsize here
    Z = cot
    if use_log:
        pcm = axs.pcolor(Z, norm=colors.LogNorm(vmin=limit[0], vmax=limit[1]), cmap='jet', shading='auto')
    else:
        pcm = axs.pcolor(Z, cmap='jet', shading='auto')
        pcm.set_clim(limit[0], limit[1])

    # Add the rectangle patch
    rectangle = patches.Rectangle((110, 85), 20, 20, linewidth=2, edgecolor='black', facecolor='none', alpha=0.9)
    axs.add_patch(rectangle)

    # axs.set_title(title, fontsize=17)
    axs.tick_params(axis='both', which='major', labelsize=17)
    axs.axes.yaxis.set_visible(True)
    axs.axes.xaxis.set_visible(True)

    cbar = fig.colorbar(pcm, ax=axs)
    cbar.ax.tick_params(labelsize=17)

    plt.savefig(fname)
    plt.close()


def plot_cot2_no(cot,title,fname,use_log,limit):
    fig,axs = plt.subplots(1,1)
    Z = cot
    if use_log:
        pcm = axs.pcolor(Z, norm=colors.LogNorm(vmin=limit[0], vmax=limit[1]),cmap='jet', shading='auto')
    else:
        pcm = axs.pcolor(Z, cmap='jet', shading='auto')
        pcm.set_clim(limit[0],limit[1])
    # axs.set_title('', fontsize=17)
    # axs.tick_params(axis='both', which='major', labelsize=17)
    axs.axes.yaxis.set_visible(False)
    axs.axes.xaxis.set_visible(False)
    # fig.colorbar(pcm, ax=axs)
    # cbar = fig.colorbar(pcm, ax=axs)
    # cbar.ax.tick_params(labelsize=17)
    # cbar.set_label( fontsize=17)
    plt.margins(x=0)
    plt.margins(y=0)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(fname,bbox_inches='tight',pad_inches=0)
    plt.close()


def plot_cmask(cmask,title,fname,limit=[0,1]):
    fig,axs = plt.subplots(1,1)
    Z = cmask
    # pcm = axs.pcolor(Z, norm=colors.LogNorm(vmin=eps, vmax=723),cmap='jet', shading='auto')
    pcm = axs.pcolor(Z, cmap='jet', shading='auto')
    pcm.set_clim(limit[0], limit[-1])
    # axs.set(title=title)
    axs.set_title(title, fontsize=17)
    axs.tick_params(axis='both', which='major', labelsize=17)
    axs.axes.yaxis.set_visible(True)
    axs.axes.xaxis.set_visible(True)
    # fig.colorbar(pcm, ax=axs)
    cbar = fig.colorbar(pcm, ax=axs)
    cbar.ax.tick_params(labelsize=17)
    plt.savefig(fname)
    plt.close()


def plot_cmask2(cmask,title,fname,limit=[0,1],colorsmap=['black','blue']):
    # custom_cmap = colors.ListedColormap(['blue', 'yellow', 'red'])
    custom_cmap = colors.ListedColormap(colorsmap)
    fig,axs = plt.subplots(1,1)
    Z = cmask
    # pcm = axs.pcolor(Z, norm=colors.LogNorm(vmin=eps, vmax=723),cmap='jet', shading='auto')
    pcm = axs.pcolor(Z, cmap=custom_cmap, shading='auto')
    pcm.set_clim(limit[0], limit[-1])
    # axs.set(title=title)
    axs.set_title(title, fontsize=17)
    axs.tick_params(axis='both', which='major', labelsize=17)
    axs.axes.yaxis.set_visible(True)
    axs.axes.xaxis.set_visible(True)
    # fig.colorbar(pcm, ax=axs)
    cbar = fig.colorbar(pcm, ax=axs)
    cbar.ax.tick_params(labelsize=17)
    plt.savefig(fname)
    plt.close()


# def plot_seaborn(x,title,fname):
#     swarm_plot = sns.heatmap(x,annot=True,cmap="Blues",vmin=0,vmax=15)
#     swarm_plot.set_xticklabels(vza)
#     swarm_plot.set_yticklabels(sza)
#     swarm_plot.set(xlabel="VZA",ylabel="SZA")
#     fig = swarm_plot.get_figure()
#     fig.savefig(fname) 
#     plt.close()


def plot_cmask3(cmask,title,fname,limit=[0,1]):
    fig,axs = plt.subplots(1,1)
    Z = cmask
    # pcm = axs.pcolor(Z, norm=colors.LogNorm(vmin=eps, vmax=723),cmap='jet', shading='auto')
    pcm = axs.pcolor(Z, cmap='plasma', shading='auto')
    pcm.set_clim(limit[0], limit[-1])
    # axs.set(title=title)
    axs.set_title(title, fontsize=17)
    axs.tick_params(axis='both', which='major', labelsize=17)
    axs.axes.yaxis.set_visible(True)
    axs.axes.xaxis.set_visible(True)
    # fig.colorbar(pcm, ax=axs)
    cbar = fig.colorbar(pcm, ax=axs)
    cbar.ax.tick_params(labelsize=17)
    plt.savefig(fname,bbox_inches='tight',pad_inches=0)
    plt.close()


def plot_cmask3_no(cmask,title,fname,limit=[0,1]):
    fig,axs = plt.subplots(1,1)
    Z = cmask
    # pcm = axs.pcolor(Z, norm=colors.LogNorm(vmin=eps, vmax=723),cmap='jet', shading='auto')
    pcm = axs.pcolor(Z, cmap='plasma', shading='auto')
    pcm.set_clim(limit[0], limit[-1])
    # axs.set(title=title)
    # axs.set_title('', fontsize=17)
    # axs.tick_params(axis='both', which='major', labelsize=17)
    axs.axes.yaxis.set_visible(False)
    axs.axes.xaxis.set_visible(False)
    # fig.colorbar(pcm, ax=axs)
    # cbar = fig.colorbar(pcm, ax=axs)
    # cbar.ax.tick_params(labelsize=17)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def plot_scatter(x,y,title,filename):
    N = x.size
    colors = np.random.rand(N)
    # area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

    # plt.scatter(x.flatten(),y.flatten(),  c=colors, alpha=0.5)
    plt.scatter(x.flatten(),y.flatten(),  alpha=0.5)
    x = [0, 7]
    y = [0, 7]
    plt.plot(x, y, color="black", linewidth=3)
    plt.xlim(0,7)
    plt.ylim(0,7)
    plt.xlabel("Ground Truth COT")
    plt.ylabel("Predicted COT")
    plt.title(title)
    plt.show()
    plt.savefig(filename)
    plt.close()


def plot_scatter2(x,y,title,filename):
    N = x.size
    colors = np.random.rand(N)
    # area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

    # plt.scatter(x.flatten(),y.flatten(),  c=colors, alpha=0.5)
    fig,ax=plt.subplots(1,1)
    ax.scatter(x.flatten(),y.flatten(),  alpha=0.5)
    # ax.xlim(-10,10)
    # plt.ylim(-10,10)
    # plt.xlabel("Ground Truth COT")
    # plt.ylabel("Predicted COT")
    # plt.title(title)
    # plt.show()
    ax.add_artist(Lines.Line2D([0,0],[1,1]))
    plt.savefig(filename)
    plt.close()

def plot_scatter3(x,y,title,filename):
    N = x.size
    colors = np.random.rand(N)
    # area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

    # plt.scatter(x.flatten(),y.flatten(),  c=colors, alpha=0.5)
    plt.scatter(x.flatten(),y.flatten(),  alpha=0.5)
    x = [0, 40]
    y = [0, 40]
    plt.plot(x, y, color="black", linewidth=3)
    plt.xlim(0, 40)
    plt.ylim(0, 40)
    plt.xlabel("Ground Truth CER")
    plt.ylabel("Predicted CER")
    plt.title(title)
    plt.show()
    plt.savefig(filename)
    plt.close()

def barplot_dice(y,delta,figtitle,filename="z33.png"):
    fig, ax = plt.subplots()

    # x = [0,1,2,3,4,5,6,7,8,9]
    x=[] 
    L=10
    Lmin=-6.6226
    Lmax=9.3373
    delta = (Lmax-Lmin)/L
    for l in range(L):
        lower_limit = Lmin+delta*l
        upper_limit = Lmin+delta*(l+1)

        x.append((upper_limit+lower_limit)/2)
    # y = np.arange(10)*20
    # bar_labels = ['red', 'blue', '_red', 'orange']
    # bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

    # ax.bar(fruits, counts, label=bar_labels, color=bar_colors)
    ax.bar(x, y)

    ax.set_ylabel('Cloud Bins')
    ax.set_ylabel('Dice Score')
    ax.set_title(figtitle)
    # ax.legend(title='Fruit color')

    plt.show()
    plt.savefig(filename)
    plt.close()

def hist_plot_COT(data,figtitle,filename='zz.png'):
    # Calculate histogram with 10 bins
    hist, bins = np.histogram(data[:], bins=10, range=(-6.6226, 9.3373))

    # Plot histogram with matplotlib 
    # Calculate bin centers
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Plot histogram at bin centers
    plt.bar(bin_centers, hist, width=1)
    # Format x-axis ticks
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plticker.FormatStrFormatter('%.2f'))
    plt.xticks(bin_centers)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(figtitle)
    plt.xlim(min(bins), max(bins))
    # plt.ylim(0, max(hist))
    plt.ylim(0, 3000)
    plt.grid(alpha=0.5)

    plt.show()
    plt.savefig(filename)
    plt.close()

def hist_plot_COT_100m(data,figtitle,filename='zz.png'):
    # Calculate histogram with 10 bins
    hist, bins = np.histogram(data[:], bins=10, range=(0, 6))

    # Plot histogram with matplotlib 
    # Calculate bin centers
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Plot histogram at bin centers
    plt.bar(bin_centers, hist, width=1)
    # Format x-axis ticks
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plticker.FormatStrFormatter('%.2f'))
    plt.xticks(bin_centers)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(figtitle)
    plt.xlim(min(bins), max(bins))
    # plt.ylim(0, max(hist))
    plt.ylim(0, 18000)
    plt.grid(alpha=0.5)

    plt.show()
    plt.savefig(filename)
    plt.close()

def normalized_hist_plot_COT_100m(data,figtitle,filename='zz.png'):
    # Calculate histogram with 10 bins
    hist, bins = np.histogram(data[:], bins=10, range=(0, 7))

    hist = hist/np.size(data)

    # Plot histogram with matplotlib 
    # Calculate bin centers
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Plot histogram at bin centers
    plt.bar(bin_centers, hist, width=1)
    # Format x-axis ticks
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plticker.FormatStrFormatter('%.2f'))
    plt.xticks(bin_centers)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(figtitle)
    plt.xlim(min(bins), max(bins))
    # plt.ylim(0, max(hist))
    plt.ylim(0, 1)
    plt.grid(alpha=0.5)

    # plt.show()
    plt.savefig(filename)
    plt.close()


def hist_plot_COT_100m_step(data,figtitle,filename='zz.png'):
    # Calculate histogram with 10 bins
    hist1, bins = np.histogram(data[:], bins=10, range=(0, 6))
    hist2, bins = np.histogram(data[:], bins=10, range=(0, 6))

    hist1 = hist1/data.size
    hist2 = hist2/data.size
    # Plot histogram with matplotlib 
    # Calculate bin centers
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    offset = 0.5
    # Plot histogram at bin centers
    plt.bar(bin_centers-offset, hist1, width=0.4)
    plt.bar(bin_centers+offset, hist2, width=0.4)

    # # Plot histogram with lines connecting bin edges
    # # plt.hist(data, bins=10, range=(0, 6), histtype='step')
    # plt.step(bin_centers,hist)
    
    # Format x-axis ticks
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plticker.FormatStrFormatter('%.2f'))
    plt.xticks(bin_centers)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(figtitle)
    plt.xlim(min(bins)-offset, max(bins)+offset)
    # plt.ylim(0, max(hist))
    plt.ylim(0, 1)
    plt.grid(alpha=0.5)

    # plt.show()
    plt.savefig(filename, dpi=600)
    plt.close()

def hist_plot_CER(data,figtitle,filename='zz.png'):
    # Calculate histogram with 10 bins
    hist, bins = np.histogram(data[:], bins=10, range=(0, 40))

    # Plot histogram with matplotlib 
    # Calculate bin centers
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Plot histogram at bin centers
    plt.bar(bin_centers, hist, width=3)
    # Format x-axis ticks
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plticker.FormatStrFormatter('%.2f'))
    plt.xticks(bin_centers)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(figtitle)
    plt.xlim(min(bins), max(bins))
    # plt.ylim(0, max(hist))
    plt.ylim(0, 3000)
    plt.grid(alpha=0.5)

    plt.show()
    plt.savefig(filename)
    plt.close()

def hist_plot_CER_100m(data,figtitle,filename='zz.png'):
    # Calculate histogram with 10 bins
    hist, bins = np.histogram(data[:], bins=10, range=(0, 40))

    # Plot histogram with matplotlib 
    # Calculate bin centers
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Plot histogram at bin centers
    plt.bar(bin_centers, hist, width=3)
    # Format x-axis ticks
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plticker.FormatStrFormatter('%.2f'))
    plt.xticks(bin_centers)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(figtitle)
    plt.xlim(min(bins), max(bins))
    # plt.ylim(0, max(hist))
    plt.ylim(0, 18000)
    plt.grid(alpha=0.5)

    # plt.show()
    plt.savefig(filename)
    plt.close()

def normalized_hist_plot_CER_100m(data,figtitle,filename='zz.png'):
    # Calculate histogram with 10 bins
    hist, bins = np.histogram(data[:], bins=10, range=(0, 40))

    hist = hist/ np.size(data)

    # Plot histogram with matplotlib 
    # Calculate bin centers
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Plot histogram at bin centers
    plt.bar(bin_centers, hist, width=3)
    # Format x-axis ticks
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plticker.FormatStrFormatter('%.2f'))
    plt.xticks(bin_centers)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(figtitle)
    plt.xlim(min(bins), max(bins))
    # plt.ylim(0, max(hist))
    plt.ylim(0, 1)
    plt.grid(alpha=0.5)

    # plt.show()
    plt.savefig(filename)
    plt.close()

def plot_joint_dist(x,y,figtitle,filename='zz2.png'):
    '''
    Plots the joint distribution, requires x and y in dim [N,2]
    '''
    # Plot 
    plt.scatter(x[:,0], x[:,1], color='red', label = "Ground Truth")
    plt.scatter(y[:,0], y[:,1], color='blue', label = "Retrievals")

    plt.xlabel('COT Values')
    plt.ylabel('CER Values')
    plt.title(figtitle)
    plt.xlim(0, 7)
    plt.ylim(0, 40)
    plt.legend()
    # plt.show()
    plt.savefig(filename)
    plt.close()

def plot_joint_dist_single(x,figtitle,filename='zz2.png',colorss='red',labels="Ground Truth"):
    '''
    Plots the joint distribution, requires x and y in dim [N,2]
    '''
    # Plot 
    # plt.scatter(x[:,0], x[:,1], color='red', label = "Groudn Truth")
    plt.scatter(x[:,0], x[:,1], color=colorss, label = labels)

    plt.xlabel('COT Values')
    plt.ylabel('CER Values')
    plt.title(figtitle)
    plt.xlim(0, 7)
    plt.ylim(0, 40)
    plt.legend()
    # plt.show()
    plt.savefig(filename)
    plt.close()

def change_dim(data):  
    new_data = np.empty([data.shape[1]*data.shape[2],2],dtype=float)
    count=0
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            new_data[count,:]= data[:,i,j]
            count+=1

    return new_data  

def change_dim_using_map(data,map):  
    new_data = np.empty([data.shape[1]*data.shape[2],2],dtype=float)
    count=0
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            if map[i,j]==1:
                new_data[count,:]= data[:,i,j]
                count+=1

    return new_data 



def plot_masks(cot, title, fname, use_log, limit):
    fig, axs = plt.subplots(1, 1)
    Z = cot
    # Define a custom colormap with three colors: blue, yellow, and red
    custom_cmap = colors.ListedColormap(['blue', 'yellow', 'red'])
    
    if use_log:
        pcm = axs.pcolor(Z, norm=colors.LogNorm(vmin=limit[0], vmax=limit[1]), cmap=custom_cmap, shading='auto')
    else:
        pcm = axs.pcolor(Z, cmap=custom_cmap, shading='auto')
        pcm.set_clim(limit[0], limit[1])
    
    axs.set_title(title, fontsize=17)
    axs.tick_params(axis='both', which='major', labelsize=17)
    axs.axes.yaxis.set_visible(True)
    axs.axes.xaxis.set_visible(True)
    
    cbar = fig.colorbar(pcm, ax=axs)
    cbar.ax.tick_params(labelsize=17)
    
    plt.savefig(fname)
    plt.close()

def plot_2d(x,y,title,fname,limit,xl,yl,font=17):
    # create the plot
    plt.plot(x,y)
    if limit is not None:
        plt.ylim(limit[0],limit[1])
    # Set the labels for x and y axes
    plt.xlabel(xl, fontsize=font)
    plt.ylabel(yl, fontsize=font)
   
    # Set the fontsize for the title, xlabel, and ylabel
    plt.title(title, fontsize=font)
    plt.xticks(fontsize=font)
    plt.yticks(fontsize=font)

     # Add grids
    plt.grid(alpha=0.5)

    plt.savefig(fname,dpi=600)
    plt.close()

def plot_relative_error_vs_true(y_true,y_pred,title,fname,limit,xl,yl,font=17):
    # Sort y_true in ascending order and get the corresponding order indices
    order_indices = np.argsort(y_true)

    # Use the order indices to sort both y_true and y_pred
    C = y_true[order_indices]
    D = y_pred[order_indices]

    # E = C-D

    # plot_2d(C,C-D,title,fname,limit,xl,yl,font)
    # create the plot
    plt.plot(C,C-D)
    # Add a black line along the x-axis where y=0
    plt.axhline(0, color='black', linestyle='--')
    if limit is not None:
        plt.ylim(limit[0],limit[1])
    # Set the labels for x and y axes
    plt.xlabel(xl, fontsize=font)
    plt.ylabel(yl, fontsize=font)
   
    # Set the fontsize for the title, xlabel, and ylabel
    plt.title(title, fontsize=font)
    plt.xticks(fontsize=font)
    plt.yticks(fontsize=font)

     # Add grids
    plt.grid(alpha=0.5)

    plt.savefig(fname,dpi=600)
    plt.close()


if __name__=="__main__":
    # Sample data
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    plot_2d(x,y,"Testing","zz.png",[0,13],"value","frequency",13)

    A1 = np.array([50, 20, 70, 10, 30])
    B1 = np.array([10, 20, 30, 40, 50])
    plot_relative_error_vs_true(A1,B1,"Testing","zz.png",None,"value","frequency",13)

    # gt1 = np.random.randint(1,440,(5,10,10,1))
    # gt2 = np.random.randint(0,2,(10,10))
    # # gt = np.concatenate((gt1,gt2),axis=3)

    # predictions = np.random.randint(-5,13,(10,10))

    # relative_plot_cot(gt2,predictions,"Relative Error Vs Ground Truth","zz.png",False,[-6,6])
    # import numpy as np

    # A = np.random.rand(5,3) 

    # B = np.random.rand(5,3)

    # A1,B1= sort_arrays(A,B)

    # print(A1)
    # print(B1)

    # # print(predictions[0,2:8,2:8,0])
    # # print(predictions)
    # zero_mask = (predictions>0)*1

    # high_mask = (predictions<=5)*1
    # # print(high_mask)

    # predictions = predictions*high_mask

    # predictions[predictions==0]=150
    # # print(predictions)

    # predictions = predictions*zero_mask

    # # print(predictions)

    # # plot_five(gt,predictions)
    # # print((gt2))
    # # plot_cot(predictions[0,:,:,0],'tush','test.png',use_log=False,limit=[0,5] )
    # limit =[0,5]
    # # print(limit[1])
        

    # # r = predictions[0,2:8,2:8,:]
    # # print(r.shape)


    # temp = np.random.randint(-5,100,(5,5))
    # # print(temp)

    # temp[temp>=50]=50
    # temp[temp<0]=0
    # # print(temp)
    # import torch
    # temp2 = torch.randint(-10,100,(1,5,5,1))
    # print(temp2)
    # temp2[temp2<0]=0
    # print(temp2)
    # print("Done !")

    # gt1 = np.random.randint(-10,10,(5,10,10,1))
    # gt2 = gt1*0.8
    # plot_scatter(x=gt1,y=gt2,title="Testing",filename="zahid2.png")

    # barplot_dice()

    
    # Generate data
    # data = np.random.randint(0, 40, 100) 
    # hist_plot(data,'Histogram of Retrieved COT')

    # Generate sample data
    # x = np.random.rand(100, 2) 
    # y = np.random.rand(100, 2)
    # x = np.random.rand(2, 70,70) 
    # y = np.random.rand(2, 70,70)
    # x = change_dim(x)
    # y = change_dim(y)
    # plot_joint_dist(x,y,'Scatter')
    gt2 = np.random.randint(0,3,(68,68))
    # plot_cmask(gt2,"Test","zzz.png",[0,1,2])