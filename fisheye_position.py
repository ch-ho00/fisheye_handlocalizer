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
    
        fig = plt.figure(figsize=(6,6),dpi=240)
        fig.set_size_inches(1, 1, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.axis('off')
        plt.imshow(img[position[1]: position[3], position[0]: position[2]])
        plt.savefig('./dataset/train_fish_hand/'+'/'+filename, dpi = 240)
        plt.close()
        
files = os.listdir('./original_dataset/aug_hand')
files.remove('handloc.csv')
files.sort()
ll = ['aug_hand']

position = []
df = pd.read_csv('./original_dataset/aug_hand/handloc.csv', header = None)
for file,l in tqdm(zip(files,df.iterrows())):                
        try:
                # divide by 2 because its reshaped
                # y pos 
                l[1][0] = l[1][0]//2
                # x pos
                l[1][1] = l[1][1]//2
                l[1][2] = 75
                
                position.append(distort(l[1][1],l[1][0])+ distort(l[1][1]+l[1][2],l[1][0]) + distort(l[1][1]+l[1][2],l[1][0]+ l[1][2])+ distort(l[1][1],l[1][0]+ l[1][2]))
                image = plt.imread('./dataset/aug_fish_hand/'+ file)
                save_fig(image,[distort(l[1][1],l[1][0])[0],distort(l[1][1],l[1][0])[1],distort(l[1][1]+l[1][2],l[1][0]+l[1][2])[0], distort(l[1][1]+l[1][2],l[1][0]+l[1][2])[1]], file)
        except:
                print(file, l[1], l[0])
                continue
position_df = pd.DataFrame(position)
position_df.to_csv('./dataset/aug_fish_hand/handloc.csv',index= False, header=False)