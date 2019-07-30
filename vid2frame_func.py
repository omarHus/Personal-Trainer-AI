import cv2
import math
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing import image
import numpy as np
from keras.utils import np_utils
from skimage.transform import resize
import os.path
from os import listdir
from os import path
import ffmpeg

def main():
    vidnames = getVidnames()
    makeFrames(vidnames,143)


def getVidnames():
    vidnames = None
    try:
        vidnames = os.listdir('videos')
    except:
        print("No files in directory")
    return vidnames
   

def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotateCode = None
    if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
        rotateCode = cv2.ROTATE_90_CLOCKWISE
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
        rotateCode = cv2.ROTATE_180
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
        rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE

    return rotateCode

def correct_rotation(frame, rotateCode):  
    return cv2.rotate(frame, rotateCode) 

def makeFrames(videoFiles, numberOfFiles):
    count = numberOfFiles
    f = open('Training/test.csv','a')
    f.write('\n')
    for videoFile in videoFiles:
        videoFile = 'videos/' + videoFile 
        cap = cv2.VideoCapture(videoFile)
        rotateCode = check_rotation(videoFile)
        while(cap.isOpened()):
            frameId = cap.get(1) #1st argument is index position of video frames
            ret, frame = cap.read()
            if (ret != True):
                break
            if (frameId % 8 == 0): #setting frame rate
                if rotateCode is not None:
                    frame = correct_rotation(frame, rotateCode)    
                filename = "image%d.jpg" % count;count+=1
                f.write(filename)
                f.write('\n')
                cv2.imwrite('Training/' + filename, frame)
        cap.release()   
    f.close()

if __name__ == "__main__":
    main()
