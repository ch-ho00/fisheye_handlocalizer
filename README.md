# Fisheye hand localizer

When there is a fisheye lense distorted image data, how does one resolve the issue of localisation? So I have posed a particular problem of attempting to localize the hand from a fisheye distorted image data.

## Result and comparison

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

    Generates a csv file in fish_hand directory | csv includes the location of vertices of distorted image

    - util.py

    Etc functions required for test and modeling

    - gesture_recognizer1.py 

    Generates plain handDetector.pkl at current directory

    ./data_augment/*
    Files for generating augmented data



 
