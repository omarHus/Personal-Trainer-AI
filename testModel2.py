# Stores the core functions for testing images against our trained model

import cv2
import math
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing import image
import numpy as np
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from skimage.transform import resize
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout
import ssl
import imageio
import ffmpeg
import os.path
from os import path
import cloudinary

def main():
    makeFrames('fakefile.txt')
    # scores = model.evaluate(test_image, test_y)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

def makeFrames(videoFile):
    frames     = []
    cap        = cv2.VideoCapture(videoFile) #use opencv to capture video file
    frameRate  = cap.get(5)
    rotateCode = check_rotation(videoFile) #check if video got rotated
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0): #only take 1/8 of the frames captured
            if rotateCode is not None:
                frame = correct_rotation(frame, rotateCode)
            frames.append(frame)
    cap.release()
    return frames

def makeFramesFromCloud(videoUrl):
    frames = []
    video_time = 0.0
    while(True):
        frame = None
        # try:
        frame = cloudinary.CloudinaryVideo(videoUrl).image(start_offset=video_time)
        frame = imageio.imread(frame)
        frames.append(frame)
        video_time += 0.1
        # except:
            # print("Error in frame upload")
            # break
        if len(frames) > 40:
            break
    return frames

#Create Gif out of labeled output images to display on results.html page
def videoOutput(frames, output_path):
    imageio.mimsave(output_path, frames, duration=0.3)
    return output_path

#Label frames based on predictions from model e.g. Good squat or bad squat
def createLabeledImages(orig_images, labels):
    font  = cv2.FONT_HERSHEY_SIMPLEX
    count = 0
    for img in orig_images:
        if labels[count] == 0: #good squat
            color = (0,255,0) #green
        else:
            color = (255,0,0) #red
        cv2.putText(img, classMap(labels[count]), (0,int(244/2)), font, 5, color) #classMap is a function i wrote to map integer of prediction to string like "Good"
        count += 1
    cv2.destroyAllWindows()
    
    return orig_images

# Load in Test Images and PreProcess if we are using csv files (not used on heroku app)
def loadInTestImages(filename):
    test = pd.read_csv(filename)
    num_of_tests = len(test)
    if hasattr(test, 'Class'):
        test_y = np_utils.to_categorical(test.Class)
    else:
        test_y = np.zeros((2,len(test)))

    test_image = []
    for img_name in test.Image_ID:
        img = plt.imread(img_name)
        test_image.append(img)
    orig_images = test_image
    test_img = np.array(test_image)

    test_image = []
    for i in range(0,test_img.shape[0]):
        a = resize(test_img[i], preserve_range=True, output_shape=(224,224)).astype(int)
        test_image.append(a)
    test_image = np.array(test_image)
    # preprocessing the images
    test_image = preprocess_input(test_image, mode='tf')

    return num_of_tests, test_y, test_image, orig_images

#Process images in format for trained model
def processImages(frames):
    num_of_frames = len(frames)
    test_y        = np.zeros((2,num_of_frames))
    orig_images   = frames
    test_img      = np.array(frames)

    test_image = []
    for i in range(0,test_img.shape[0]):
        a = resize(test_img[i], preserve_range=True, output_shape=(224,224)).astype(int)
        test_image.append(a)
    test_image = np.array(test_image)
    # preprocessing the images
    test_image = preprocess_input(test_image, mode='tf')

    return num_of_frames, test_y, test_image, orig_images

# extracting features from the images using pretrained model
def load_basemodel(testImages, num_of_tests):
    #import vgg16 base model
    if (not os.environ.get('PYTHONHTTPSVERIFY', '') and 
        getattr(ssl, '_create_unverified_context', None)): 
        ssl._create_default_https_context = ssl._create_unverified_context
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    testImages = base_model.predict(testImages)
    # converting the images to 1-D form
    testImages = testImages.reshape(num_of_tests, 7*7*512)

    # zero centered images
    testImages = testImages/testImages.max()
    return testImages

def loadTrainedModel(modelName):
    #Load in Trained Model
    loaded_model = Sequential()
    loaded_model.add(InputLayer((7*7*512,)))    # input layer
    loaded_model.add(Dense(units=1024, activation='sigmoid')) # hidden layer
    loaded_model.add(Dense(2, activation='softmax'))    # output layer
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    loaded_model.load_weights(modelName)
    return loaded_model

def makepredictions(modelName, testCases):
    predictions = modelName.predict_classes(testCases)
    return predictions

#Checking to see if video file got rotated
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

def classMap(classNumber):
    if classNumber == 0:
        return "Good Squat"
    else:
        return "Bad Squat"


if __name__ == "__main__":
    main()