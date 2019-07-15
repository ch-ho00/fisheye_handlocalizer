# Model that detects the hand location when given a fisheye image data
#!/usr/bin/env python
from utils import *
from data_process import *
from gesture_recognizer1 import * 
import os
from tqdm import tqdm
import cv2
import pandas as pd 
import matplotlib.pyplot as plt
def dir2hog(x):
    '''
    Converts list of train image directories into hog form of cropped image
    If image is hand then crops hand
    else crops random position of image with side 50
    Input : x (dir, vertice_locations)
    Output: hog-version x
    '''
    
    x_hog = []
    for (i,j) in x:
        img = plt.imread(i)
        try:
            if len(j) == 8:
                tmp = crop2(img, j)
            elif len(j) == 3:
                # j = (xtop, ytop, side)
                tmp = img[j[1]:j[1]+j[2], j[0]:j[0]+j[2]]
        except:
            side = 70
            x_rand = random.randint(0,240-side)
            y_rand = random.randint(0,240-side) 
            tmp = crop2(img, [x_rand, y_rand, x_rand+side, y_rand+side])
        tmp = resize(tmp,((150,150)))
        tmp = rgb2gray(tmp)
        x_hog.append(hog(tmp)) 
    return x_hog

def img2hog(x):
    '''
    Converts list of train image directories into hog form of cropped image
    If image is hand then crops hand
    else crops random position of image with side 50
    Input : x (np.ndarray , vertice_locations [xtop,ytop,side])
    Output: hog-version x
    '''
    img = x[0]
    try:
        if len(x[1]) == 8:
            tmp = crop2(img, x[1])
        elif len(x[1]) == 3:
            # j = (xtop, ytop, side)
            tmp = crop2(img,[x[1][0],x[1][1],x[1][0]+x[1][2], x[1][1]+x[1][2]])
    except:
        side = 70
        x_rand = random.randint(0,240-side)
        y_rand = random.randint(0,240-side) 
        tmp = crop2(img, [x_rand, y_rand, x_rand+side, y_rand+side])
    tmp = resize(tmp,((150,150)))
    tmp = rgb2gray(tmp)
    return hog(tmp)

def concat_boxes(boxes, overlap_thresh):
    '''
    Given different boxes each box indicating the maximum probability for a particular scale,
    the function returns the box with maximum probability of hand existing
    Input:  boxes (x_top, y_top, x_bot, y_bot, probability)
            overlap_thresh - proportion of overlap allowed for two different box with different probability
    Output: final box 
    '''
    print("concat box")
    print(boxes)
    if len(boxes) == 0:
        return []
    # convert to float
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    x1 = boxes[:,0]
    x2 = boxes[:,1]
    y1 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]
    
    # sort the boxes according to probability
    ss = np.argsort(s)
    area = (x2-x1 +1)*(y2-y1 +1)
    # list for keeping track of boxes to choose
    pick = []

    while len(ss) > 0:
        counter = len(ss) -1
        index = ss[counter]
        pick.append(index)

        xx1 = np.maximum(x1[index], x1[ss[:counter]])
        yy1 = np.maximum(y1[index], y1[ss[:counter]])
        xx2 = np.minimum(x2[index], x2[ss[:counter]])
        yy2 = np.minimum(y2[index], y2[ss[:counter]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[ss[:counter]]

        # delete all indexes from the index list that have low overlap
        ss = np.delete(ss, np.concatenate(([counter],
            np.where(overlap > overlap_thresh)[0])))

    return boxes[pick]

def predict_w_scale(image,model, scale,side):
    '''
    Input:  Image - test image
            model - handDetector model
            scale 
    Output: Box with maximum probability under the scale
    '''
    max_prob = -1
    image_np = plt.imread(image)
    rescaled_image = rescale(image_np,scale)
    
    out_box = []
    max_x = rescaled_image.shape[1]
    max_y = rescaled_image.shape[0]
    x_step = 32
    y_step = 24
    for x in range(0,max_x-side, x_step):
        for y in range(0,max_y - side, y_step):
            img_hog = img2hog((rescaled_image, [x,y,side]))
            # print(img_hog.shape)
            prob = model.predict_proba([img_hog])
            
            if prob[0][1] > max_prob:
                max_prob = prob[0][1]
                out_box = [x,y,prob[0][1],scale]
    return out_box

def draw_box(image, model, save_dir, filename):
    ''' 
    Save image with box on model's prediction
    process involves predicting boxes from different scale then aggregating them
    Input: image, hand-detector model, save_directory, filename
    Output: boxed-image at save_dir
    '''
    print("Draw Box")
    scales = [   1.25,
            1.015625,
            0.78125,
            0.546875,
            1.5625,
            1.328125,
            1.09375,
            0.859375,
            0.625,
            1.40625,
            1.171875,
            0.9375,
            0.703125,
            1.71875,
            1.484375
            ]
    pred_boxes = [] # [x,y,scale]
    side= 300
    for s in scales:
        # pred_boxes contains array of [x,y,prob, scale]
        pred_boxes.append(predict_w_scale(image, model, s,side))
    sides = [side/scales[i] for i in range(len(scales))]
    pred_boxes = [[box[0]/scales[i], box[1]/scales[i], box[2], box[3]] for i,box in enumerate(pred_boxes) ]
    boxes =[]
    for i in range(len(pred_boxes)):
        # x1, y1, x2, y2, prob, scale
        boxes.append([pred_boxes[i][0],pred_boxes[i][1],pred_boxes[i][0]+ sides[i], pred_boxes[i][1]+ sides[i],pred_boxes[i][2],scales[i]]) 
    # (N,4) 
    boxes = np.array(boxes)
    # TODO: find optimal overlap_threshold 
    final = concat_boxes(boxes, 0.4)
    x_top = int(final[0][0]/ final[0][5])
    y_top = int(final[0][1] / final[0][5])
    side_img= int((final[0][2] - final[0][0])/final[0][5])
    image_np = plt.imread(image)
    print(image_np.shape)
    print('\n\n\n')
    print(final)
    print('\n\n\n')
    
    print(y_top, x_top, side)
    image_np = np.array(image_np)
    plt.imshow(image_np[y_top: y_top+side_img,x_top: x_top+side_img])
    plt.savefig(save_dir+'/'+str(side)+"_"+filename)
    # save_fig(image_np,[x_top, x_top+side, y_top, y_top+side])    

def overlap(args1, args2):
    '''
    Calculate the overlapping area of two boxes
    Input : args1 - tuple (x_top, y_top, side)
            args2 - tuple (x_top, y_top, side)
    Output: proportion of overlapping area to total area of two boxes
    '''
    x1_tl = args1[0]
    x2_tl = args2[0]
    x1_br = args1[0] + args1[2]
    x2_br = args2[0] + args2[2]
    y1_tl = args1[1]
    y2_tl = args2[1]
    y1_br = args1[1] + args1[2]
    y2_br = args2[1] + args2[2]
    # Calculate the overlapping Area
    x_overlap = max(0, min(x1_br, x2_br)-max(x1_tl, x2_tl))
    y_overlap = max(0, min(y1_br, y2_br)-max(y1_tl, y2_tl))
    overlap_area = x_overlap * y_overlap
    area_1 = args1[2] * args2[2]
    area_2 = args2[2] * args2[2]
    total_area = area_1 + area_2 - overlap_area
    return overlap_area / float(total_area)

def evaluate(x, label, model, step_x=20, step_y=20):
    '''
    Iteratively train model by checking random crop of image
    If the prediction is wrong then will be returned to be trained further at hard_negative_mining

    Input : x - tuple of (train image directories, vertices) 
            label - label of train image
            model - localizer to be trained
            step_x - step in x direction
            step_y - step in y direction
            
    Output : x_hog - hog which model failed to predict correctly
            label -  label of false predicted hog (i.e. list of 0s ) 
            false_pos - # false positive in prediction
    '''
    out_hog = []
    label = []
    false_pos = 0 
    for img_dir, locs in x:
        # locs turns into x_min, y_min, side of true value
        locs = points2minmax(locs)
        arg1 = locs
        img_np = plt.imread(img_dir)
        shape = img_np.shape
        for randx in range(0,shape[1]-locs[2], step_x):       
            for randy in range(0, shape[0]-locs[2], step_y): 
                arg2 = [randx,randy, locs[2]]
                z = overlap(arg1,arg2)
                img_np2 = img_np[randy:randy+locs[2], randx: randx +locs[2]]
                img_np2 = resize(img_np2, ((150,150)))
                img_hog = hog(rgb2gray(img_np2))
                predicted = model.predict([img_hog])[0]
                if predicted == 1 and z < 0.5:
                    out_hog.append(img_hog)
                    false_pos += 1 
                if false_pos > 60000:
                    label = [0] * false_pos
                    return out_hog, label, false_pos    
    label = [0] * false_pos
    return out_hog, label, false_pos   

def hard_negative_mining(x, label,threshold= 601, max_iter = 50):
    '''
    Train model through hard negative mining
    Input : x - (train image directories, coordinates of hand)
            label - corresponding labels
            threshold - number of false-positives 
            max_iter - number of iteration
    Output : trained model 
    '''
    model = MultinomialNB()
    i = 0
    x_hog = dir2hog(x) 
    print("XXXXXXXXXXXXXXX",len(x_hog), x_hog[0].shape)
    while 1:
        i+= 1
        model = model.partial_fit(x_hog,label, classes=[0,1])
        x_hog, label, fal_pos = evaluate(x, label, model)
        print("Number of false-positives", fal_pos,x_hog[0].shape,len(x_hog))
        if fal_pos == 0:
            return model
        elif fal_pos < threshold:
            return model
        elif i > max_iter:
            return model


class hand_localizer(object):
    def __init__(self, data_director='./dataset/fish_hand'):
        '''
        train_data : Tuple (directory to train image, hand_vertice_coordinates (if not hand then none)) 
        train_label : Binary of whether hand or not hand
        handDetector : model
        '''
        self.train_data, self.train_label = train_dirs(data_director)
        self.handDetector = None
        self.test_data = None

    def train(self):
        print("Training start")
        self.handDetector = hard_negative_mining(self.train_data, self.train_label)
        print("Training end")
    def save(self,filename):
        '''
        save model under models directory
        '''
        pickle.dump(self.handDetector,gzip.open('./models/'+filename,'wb'))    
    def evaluate(self,save_dir):
        '''
        evaluate model and save results in results directory
        '''
        test_imgs = os.listdir('./dataset/test_data')
        for img in tqdm(test_imgs):
            draw_box('./dataset/test_data/'+img, self.handDetector,save_dir, img)
    def further_train(self, train_dir='./dataset/error'):
        '''
        By moving model's false output to './dataset/error' the model further trains on the false inputs
        '''
        # augment more hand data
        test_imgs = os.listdir('./dataset/train_fish_hand')
        for i,img in enumerate(test_imgs):
            if i % 1000 == 0:
                print(i)
            img_np = plt.imread('./dataset/train_fish_hand/'+img)
            img_np = resize(img_np, ((150,150)))
            img_hog = hog(rgb2gray(img_np))                        
            self.handDetector.partial_fit([img_hog], [1])

        # augment more other dataset with false label
        # test_imgs = os.listdir('./dataset/error')
        # for i,img in enumerate(test_imgs):
        #     img_np = plt.imread('./dataset/error/'+img)
        #     print(i,img, img_np.shape)
            
        #     if img_np.shape[0] + img_np.shape[1] < 250:
        #         img_np = resize(img_np, ((150,150)))
        #         img_hog = hog(rgb2gray(img_np))                        
        #         self.handDetector.partial_fit([img_hog], [0])
        #     else:
        #         for x in range(0,img_np.shape[1]- 100,24):
        #             for y in range(0,img_np.shape[0]- 100,24):
        #                 img_np2 = img_np[y: y+100 , x: x+ 100]
        #                 img_np2 = resize(img_np2, ((150,150)))
        #                 img_hog = hog(rgb2gray(img_np2))
        #                 self.handDetector.partial_fit([img_hog], [0])
# pretrain_model = "0712_aug.pkl"
# model = hand_localizer()
# if pretrain_model in os.listdir('./models/'):
#     model.handDetector = pickle.load(gzip.open('./models/'+pretrain_model, 'rb'))
# else:
#     model.train()
# model.save("0712_aug.pkl")
# model.evaluate("./results/fisheye_detector/test_data")
