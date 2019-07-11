import os
from tqdm import tqdm
import cv2
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def save_fig(img,position):
    '''
    save img with detected hand location in square
    Input:  img - npnd array
            position - [x1,y1,x2,y2]
    Output: saved image in directory
    '''
    
    rect = patches.Rectangle((position[0],position[1]),position[2]- position[0],position[3]- position[1],linewidth=3,edgecolor='r',facecolor='none')
    fig,ax = plt.subplots(1)
    ax.imshow(img)
    ax.add_patch(rect)
    plt.savefig('result.png')


def train_dirs(dir):
    files = os.listdir(dir)
    files.sort()
    x_hand = []
    x = []
    y = []
    for file in files:
        imgs = os.listdir(dir+'/'+file)
        try:
            imgs.remove(file+'.csv')
            imgs.sort()
            positions = pd.read_csv(dir+'/'+file+'/'+file + '.csv', header=None)
            for img,pos in zip(imgs,positions.iterrows()):
                x_hand.append((dir+'/'+file+'/'+img, pos[1].values))
                # x_hand = dir2hog(x_hand)
                y.append(1)
        except:
            for img in imgs:
                x.append((dir+'/'+file+'/'+img, None))
                y.append(0)
                # x = dir2hog(x)
    x = x_hand +  x
    return x,y