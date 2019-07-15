from skimage import transform, data, io
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
from random import randint
from tqdm import tqdm
from skimage.transform import resize


rand = [] 
imgs = os.listdir('./original_dataset/hand/')
imgs.sort()
for img in tqdm(imgs):
    if 1:
        zeros = np.zeros((480,480))
        hand = plt.imread('./original_dataset/hand/'+img)
        hand = hand[:, 150:450]
        hand = resize(hand, ((150,150)))
        rand_x = randint(0,480-hand.shape[0])
        rand_y = randint(0,480-hand.shape[1])
        rand.append([rand_x,rand_y])
        zeros[rand_x:rand_x+hand.shape[0], rand_y: rand_y + hand.shape[1]] = hand
        # save image
        fig = plt.figure(figsize=(6,6),dpi=240)
        fig.set_size_inches(1, 1, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.axis('off')
        plt.imshow(zeros/255)
        plt.savefig('./original_dataset/aug_hand/'+'/'+img)
        plt.close()
    else:
        print("err", img)
        continue
rand_df = pd.DataFrame(rand)
rand_df['side'] = 120
rand_df.to_csv('./original_dataset/aug_hand/handloc.csv',index=False,header=False)
