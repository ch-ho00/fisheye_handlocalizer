## Fisheye image hand localisation model

preprocess.py 
    generates cropped hand in ./dataset/cropped_hand and a csv file in cropped_hand directory | it makes use of the orignal dataset | includes the side of the cropped hand
    ### Remark
        output of the cropped image is resized to 128 *128

data_aug.py
    generates image of cropped hand at random location makes use of images from ./dataset/cropped_hand 
    along with a csv file in ./dataset/augmented directory | csv includes x,y position along with hand 


fisheye.py
    apply fisheye distortion to image


fisheye_position.py
    generates a csv file in fish_hand directory | csv includes the location of vertices of distorted image
    ### Remark
        location of vertices are in order of top left to clockwise

gesture_recognizer1.py
    generates handDetector.pkl at current directory

localizer.py


pretrained_undistorted.py
    Using pretrained model, crop the hand from an image and save cropped image to ./dataset/cropped_hand


 
