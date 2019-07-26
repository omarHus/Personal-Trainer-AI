import cv2
import math
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing import image
import numpy as np
from keras.utils import np_utils
from skimage.transform import resize
import os.path
from os import path
from enum import Enum
import ffmpeg

class ModelMode(Enum):
    TRAINING = 1
    TESTING  = 2

def main():
    vidname   = getVidname()
    makeFrames(vidname,'Training')

def mode():
    while(True):
        modenum = int(input("Enter Mode:\n1 - Training\n2 - Testing\n"))
        print(modenum)
        if (modenum == 1 or modenum == 2):
            break
        else:
            print("Incorrect Input")
    return modenum

def getVidname():
    while(True):
        vidname = input("Enter Filename\n")
        if (path.exists(vidname)):
            break
        else:
            print("Incorrect Input")
    return vidname  

def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)
    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotateCode = None
    try:
        if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
            rotateCode = cv2.ROTATE_90_CLOCKWISE
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
            rotateCode = cv2.ROTATE_180
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
            rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE
    except KeyError as error:
        rotateCode = None
    
    return rotateCode

def correct_rotation(frame, rotateCode):  
    return cv2.rotate(frame, rotateCode) 

def makeFrames(videoFile, directory):
    cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
    rotateCode = check_rotation(videoFile)
    frameRate = cap.get(5) #frame rate
    count = 0
    csv_file = directory + "/newImports.csv"
    f = open(csv_file,'w')
    f.write("Image_ID\n")
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            if rotateCode is not None:
                frame = correct_rotation(frame, rotateCode)    
            filename = directory + "/image%d.jpg" % count;count+=1
            f.write(filename)
            f.write('\n')
            cv2.imwrite(filename, frame)
    cap.release()
    f.close()
    print("Frames made successfully!")

if __name__ == "__main__":
    main()
