from skimage import transform, data, io
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
from random import randint
from tqdm import tqdm
from skimage.transform import resize


files = os.listdir('./original_dataset/cropped_hand/')
for file in files:
    rand = []
    imgs = os.listdir('./dataset/cropped_hand/'+file)
    imgs.sort()
    print(imgs)
    for img in tqdm(imgs):
        print(img)
        try:
            zeros = np.zeros((480,480,3))
            hand = plt.imread('./dataset/cropped_hand/'+file+'/'+img)
            rand_x = randint(0,480-hand.shape[0])
            rand_y = randint(0,480-hand.shape[1])
            rand.append([rand_x//2,rand_y//2])
            zeros[rand_x:rand_x+hand.shape[0], rand_y: rand_y + hand.shape[1],:] = hand
            # save image
            fig = plt.figure(figsize=(6,6),dpi=240)
            fig.set_size_inches(1, 1, forward=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            plt.axis('off')
            plt.imshow(zeros/255)
            plt.savefig('./dataset/augment/'+file+'/'+img)
        except:
            continue
        print(rand)
        exit()
    rand_df = pd.DataFrame(rand)
    side = pd.read_csv('./dataset/cropped_hand/'+file+'/'+file+'.csv', header = None)
    rand_df['side'] = side
    rand_df.to_csv('./dataset/augment/'+file+'/'+file+'.csv',index=False,header=False)
