## Model and result/comparison

Refer to the notebook in the directory.

## Files 

- localizer.py
    fisheye localizer script involves hard negative mining 

- preprocess.py 
    generates cropped hand in ./dataset/cropped_hand and a csv file in cropped_hand directory | it makes use of the orignal dataset | includes the side of the cropped hand

- pretrained_undistorted.py
    Using pretrained model, crop the hand from an image and save cropped image to ./result/...


- pretrained_undistorted.py
    Using pretrained model, crop the hand from an image and save cropped image to ./dataset/cropped_hand

- fisheye_position.py
    generates a csv file in fish_hand directory | csv includes the location of vertices of distorted image
    
- util.py
    etc functions required for test and modeling

- gesture_recognizer1.py
    generates plain handDetector.pkl at current directory




 
