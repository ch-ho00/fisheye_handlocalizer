import os
from gesture_recognizer1 import *
import cv2 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
from tqdm import tqdm
import pandas as pd 
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

def draw_box(model,img, filename, dir):
    scales = [   1.25, 1.015625, 0.78125, 0.546875, 1.5625, 1.328125, 1.09375, 0.859375, 0.625, 1.40625, 1.171875, 0.9375, 0.703125, 1.71875, 1.484375]
    detectedBoxes = [] ## [x,y,conf,scale]
    sid= 500
    for sc in scales:
        print(sc)
        detectedBoxes.append(image_pyramid_step(model,img,sid,scale=sc))
    side = [0 for i in range(len(scales))]
    for i in range(len(scales)):
        side[i]= sid/scales[i] 
    for i in range(len(detectedBoxes)):
        detectedBoxes[i][0]=detectedBoxes[i][0]/scales[i] #x
        detectedBoxes[i][1]=detectedBoxes[i][1]/scales[i] #y

    nms_lis = [] #[x1,x2,y1,y2]
    for i in range(len(detectedBoxes)):
        nms_lis.append([detectedBoxes[i][0],detectedBoxes[i][1],
                        detectedBoxes[i][0]+side[i],detectedBoxes[i][1]+side[i],detectedBoxes[i][2]])
    nms_lis = np.array(nms_lis)

    res = non_max_suppression_fast(nms_lis,0.4)

    output_det = res[0]
    x_top = output_det[0]
    y_top = output_det[1]
    side = output_det[2]-output_det[0]
    position = [x_top, y_top, x_top+side, y_top+side]
    
    height = width = side
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.axis('off')
    plt.imshow(img[position[1]:position[3]+20,position[0]: position[2]+20])
    plt.savefig('./results/fisheye_detector/0709_side'+str(sid)+'/augg_'+filename)
    print(x_top, y_top, side)
    return x_top, y_top, side 

files = os.listdir('./dataset')
model =  pickle.load(gzip.open("./models/0709_aug.pkl", 'rb'))

for file in files:
    if file == "cropped_hand":
        continue
    elif "test_data" in file:
        imgs = os.listdir('./dataset/'+file)
        for img in tqdm(imgs):
            if 1:
                img_np = cv2.imread('./dataset/'+file+'/'+img)
                x ,y , side =draw_box(model,img_np,img,file)
            else:
                print("err")
                continue
    