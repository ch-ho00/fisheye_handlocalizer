import os
from tqdm import tqdm
import cv2
import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.patches as patches


def points2minmax(points):
    '''
    Input : list of numbers ( which are actually collections of (x,y) vertices)
    Output: x_min, y_min, side
    '''
    x_max = 0
    x_min = 1000 
    y_max = 0
    y_min = 1000 
    for count, point in enumerate(points):
        if count % 2 == 0 and point > x_max:
            x_max = point
        if count % 2 == 0 and point < x_min:
            x_min = point
        if count % 2 == 1 and point > y_max:
            y_max = point
        if count % 2 == 1 and point < y_min:
            y_min = point
    return [x_min, y_min, max(x_max-x_min, y_max-y_min)]
def crop2(img , pos):
    '''
    Crops images based on the location of vertices
    '''
    x_max = 0
    x_min = 1000 
    y_max = 0
    y_min = 1000 
    for count, point in enumerate(pos):
        if count % 2 == 0 and point > x_max:
            x_max = point
        if count % 2 == 0 and point < x_min:
            x_min = point
        if count % 2 == 1 and point > y_max:
            y_max = point
        if count % 2 == 1 and point < y_min:
            y_min = point
    img = img[y_min:y_max, x_min:x_max]
    return img
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