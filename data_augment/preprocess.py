# generate csv file containing side of cropped hand
import pandas as pd
import os
from skimage.transform import resize
from skimage import io
import matplotlib.pyplot as plt
from tqdm import tqdm
#takes returns cropped image 
def crop(img,x1,x2,y1,y2):
    crp=img[y1:y2,x1:x2]
    crp=resize(crp,((128,128)))#resize
    return crp

files = os.listdir('./dataset/')
for file in files:
    if 'user' in file:
        imgs = os.listdir('./dataset/'+file)
        l = pd.read_csv('./dataset/'+file+'/'+file+'_loc.csv')
        l['side'] = l['bottom_right_y'] - l['top_left_y']
        l['side'].to_csv('./dataset/cropped_hand/'+file+'/'+file+'.csv',index= False, header= False) 
        imgs.remove(file+'_loc.csv')
        for img,locs in tqdm(zip(imgs,l.iterrows())):
                try:
                        tmp = plt.imread('./dataset/'+file+'/'+img)
                        tmp = crop(tmp, l['top_left_x'].values[0], l['bottom_right_x'].values[0]+10, l['top_left_y'].values[0],l['bottom_right_y'].values[0]+10)
                        fig = plt.figure()
                        fig.set_size_inches(1, 1, forward=False)
                        ax = plt.Axes(fig, [0., 0., 1., 1.])
                        ax.set_axis_off()
                        fig.add_axes(ax)
                        plt.axis('off')
                        plt.imshow(tmp)
                        plt.savefig('./dataset/cropped_hand/'+file+'/'+img)
                except:
                        continue
                