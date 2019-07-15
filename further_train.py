from utils import *
from data_process import *
from localizer import * 
import os
from tqdm import tqdm
import cv2
import pandas as pd 
import matplotlib.pyplot as plt

pretrain_model = "0711_aug.pkl"
model = hand_localizer()
model.handDetector = pickle.load(gzip.open('./models/'+pretrain_model, 'rb'))
model.further_train()
# model.evaluate("./results/fisheye_detector/test_data")
model.save("further3_0711_aug.pkl")