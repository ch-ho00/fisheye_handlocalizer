# this file is to convert the coordinate locations of hands into distorted hand locations 
import pandas as pd
import numpy as np
from tqdm import tqdm
import skimage
from skimage.transform import rescale
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
def fisheye(xy):
    '''
    Fisheye transformation function | output is mapped to the original input through skimage.transform.warp
    input : (786400,2) size xy-coordinates of distorted image 
    output: (786400,2) size xy-coordinates of orignal image 
    '''
    center = np.mean(xy, axis=0)
    xc, yc = (xy - center).T
    # Polar coordinates
    r = np.sqrt(xc**2 + yc**2)
    theta = np.arctan2(yc, xc)

    r = 0.8 * np.exp(r**(1/2.1) / 1.8)
    return np.column_stack((
        r * np.cos(theta), r * np.sin(theta)
        )) + center

def shift_down(xy):
    '''
    Test transformation, translates image downwards by 100 pix 
    '''
    # print(xy[3200:3210])
    xy[:, 1] -= 100
    # print(xy[3200:3210])
    return xy

def distort(x,y):
    '''
    Inverse transformation of fisheye distortion
    input : x,y coordinate of original image
    output: x,y coordiante of distorted image
    '''
    x -= 119.5
    y -= 119.5 
    the = np.arctan2(y,x)
    r = np.sqrt(x**2 + y**2)
    r = (1.8*np.log(1.25*r))**2.1
    return [int(r*np.cos(the)+119.5), int(r*np.sin(the)+119.5)]

def save_fig(img,position, filename):
    '''
    save img with detected hand location in square
    Input:  img - npnd array
            position - [x1,y1,x2,y2]
            filename
    Output: saved image in directory
    '''
    
    rect = patches.Rectangle((position[0],position[1]),position[2]- position[0],position[3]- position[1],linewidth=3,edgecolor='r',facecolor='none')
    fig,ax = plt.subplots(1)
    ax.imshow(img)
    ax.add_patch(rect)
    plt.savefig('./dataset/patch_fish/'+filename)


files = os.listdir('./dataset//')
ll = ['user_3','user_6','user_7','user_9','user_10','user_4','user_5']

for file in files:
        if file not in ll:
                continue
        else:
                position = []
                df = pd.read_csv('./dataset/'+file+'/'+file+'_loc.csv', header = None)
                for i, l in tqdm(df.iterrows()):
                        if i == 0:
                                continue
                        l[1] = int(l[1])
                        l[2] = int(l[2])
                        l[3] = int(l[3])
                        l[4] = int(l[4])
                        
                        position.append(distort(l[1],l[2])+ distort(l[3],l[2]) + distort(l[3],l[4])+ distort(l[1],l[4]))
                        image = plt.imread('./dataset/fish_hand/'+l[0])
                        save_fig(image,[distort(l[1],l[2])[0],distort(l[1],l[2])[1],distort(l[3]+5,l[4])[0], distort(l[3]+5,l[4]+5)[1]],l[0])
                position_df = pd.DataFrame(position)
                position_df.to_csv('./dataset/fish_hand/'+file+'/'+file+'.csv',index= False, header=False)