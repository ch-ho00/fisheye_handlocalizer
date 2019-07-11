from skimage import transform, data, io
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm                 
def fisheye(xy):
    center = np.mean(xy, axis=0)
    xc, yc = (xy - center).T
    # Polar coordinates
    r = np.sqrt(xc**2 + yc**2)
    theta = np.arctan2(yc, xc)

    r = 0.8 * np.exp(r**(1/2.1) / 1.8)

    return np.column_stack((
        r * np.cos(theta), r * np.sin(theta)
        )) + center
l = ['user_3','user_6','user_7','user_9','user_10','user_4','user_5']
dirs = os.listdir('./dataset/')
for dir in dirs:
        if dir not in l:
                continue
        imgs = os.listdir('./dataset/'+dir+'/')
        for img in tqdm(imgs):
                try:
                        image = io.imread('./dataset/'+dir+'/'+img)
                        out = transform.warp(image, fisheye)
                        sizes = out.shape
                        height = float(sizes[0])
                        width = float(sizes[1])
                        fig = plt.figure(figsize=(6,6),dpi=240)
                        fig.set_size_inches(width/height, 1, forward=False)
                        ax = plt.Axes(fig, [0., 0., 1., 1.])
                        ax.set_axis_off()
                        fig.add_axes(ax)
                        plt.axis('off')
                        plt.imshow(out)
                        plt.savefig('./dataset/fish_hand/'+dir+'/'+img, dpi = height)
                except:
                        print("err  ",img)
                        continue
