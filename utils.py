import os
from tqdm import tqdm
import cv2
import pandas as pd 
import matplotlib.pyplot as plt
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
