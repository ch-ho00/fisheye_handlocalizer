import numpy as np
import skimage
from skimage.transform import rescale
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
def fisheye(xy):
    '''
    Fisheye transformation function | output is mapped to the original input through skimage.transform.warp
    input : (786400,2) size xy-coordinates of distorted image 
    output: (786400,2) size xy-coordinates of orignal image 
    '''
    center = np.mean(xy, axis=0)
    xc, yc = (xy - center).T
    print(center)
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

def restore(x,y):
    '''
    Inverse transformation of fisheye distortion
    input : x,y coordinate of original image
    output: x,y coordiante of distorted image
    '''
    x -= 159.5
    y -= 119.5 
    the = np.arctan2(y,x)
    r = np.sqrt(x**2 + y**2)
    r = (1.8*np.log(1.25*r))**2.1
    print(r,the)
    print(r*np.cos(the)+159.5,r*np.sin(the)+119.5)

img = skimage.io.imread('./dataset/user_3/A2.jpg')
print("XXXXXXXXXX",img[0][0])
conv = skimage.transform.warp(img,fisheye)

# coordinate in distorted image
coordin = [125,200]
coordin[0] -= 159.5
coordin[1] -= 119.5
r = np.sqrt(coordin[0]**2 + coordin[1]**2)
theta = np.arctan2(coordin[1], coordin[0])
r = 0.8 * np.exp(r**(1/2.1) / 1.8)
# coordinate in original image
print((r * np.cos(theta)+ 159.5, r * np.sin(theta)+ 119.5))    

# conv2 = skimage.transform.warp(img,shift_down)
# plt.imshow(conv2)
# plt.savefig('./conv2.png')
# print(img.shape,conv2.shape)
# print(conv2[80][13],img[1][13])

# print(img[198][126])
# print(conv[200][125])
# plt.figure(1)
# plt.imshow(img)
# plt.scatter([126],[198])
# plt.savefig('./original.png')

# plt.figure(2)
# plt.imshow(conv)
# plt.scatter([125],[200])
# plt.savefig('./fish.png')
