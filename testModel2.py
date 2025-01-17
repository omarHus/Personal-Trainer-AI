# Stores the core functions for testing images against our trained model

import cv2
import math
import pandas as pd
from keras.preprocessing import image
import numpy as np
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from skimage.transform import resize
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout
import ssl
import imageio
import ffmpeg
import os.path
from os import path
import cloudinary
import cloudinary.uploader

def main():
    testFrames = makeFrames('videos/VID_20190805_232228_ualszf.mp4')
    origFrames = testFrames
    testFrames = processImages(testFrames)
    base_model = load_basemodel()
    testFrames = classifyImages(testFrames,base_model)
    weights    = get_file('trained_model.h5','https://github.com/omarHus/physioWebApp/raw/master/trained_model.h5')
    model      = loadTrainedModel(weights)
    prediction = makepredictions(testFrames,model)
    testFrames = createLabeledImages(origFrames,prediction)
    output     = videoOutput(testFrames,'output.mp4')
    fileID     = videoOut2Cloud('output.mp4', "omar1")
    imgtag     = cloudinary.CloudinaryVideo(fileID, format="mp4").video()
    print("ImgTag is ",imgtag)
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

    return test_image

# extracting features from the images using pretrained model
def load_basemodel():
    #import vgg16 base model
    if (not os.environ.get('PYTHONHTTPSVERIFY', '') and 
        getattr(ssl, '_create_unverified_context', None)): 
        ssl._create_default_https_context = ssl._create_unverified_context
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model._make_predict_function()
    return base_model

def classifyImages(frames, base_model):
    num_of_frames = len(frames)
    frames = base_model.predict(frames, batch_size=num_of_frames)
    # # converting the images to 1-D form
    frames = frames.reshape(num_of_frames, 7*7*512)
    # # zero centered images
    frames = frames/frames.max()
    return frames

def loadTrainedModel(weights_path):
    #Load in Trained Model
    loaded_model = Sequential()
    loaded_model.add(InputLayer((7*7*512,)))    # input layer
    loaded_model.add(Dense(units=1024, activation='sigmoid')) # hidden layer
    loaded_model.add(Dense(2, activation='softmax'))    # output layer
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    loaded_model.load_weights(weights_path)
    loaded_model._make_predict_function()
    return loaded_model

def makepredictions(testCases, modelName):
    myPredictions = modelName.predict_classes(testCases)
    return myPredictions

#Label frames based on predictions from model e.g. Good squat or bad squat
def createLabeledImages(orig_images, labels):
    font      = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 5
    line_type = 2
    count = 0
    for img in orig_images:
        if labels[count] == 0: #good squat
            color = (0,255,0) #green
        else:
            color = (255,0,0)

        cv2.putText(img, classMap(labels[count]), (0, int(244/2)), font, fontScale, color, line_type) #classMap is a function i wrote to map integer of prediction to string like "Good"
        count += 1
    cv2.destroyAllWindows()
    return orig_images   #Create Gif out of labeled output images to display on results.html page
def videoOutput(frames, output_path):
    # imageio.mimsave(output_path, frames, duration=0.3) #this makes a gif
    height, width, layers = frames[0].shape
    size                  = (width,height)

    # set up videowriter in opencv
    fourcc   = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps      = 1.1
    videoOut = cv2.VideoWriter(output_path, fourcc, fps, size)

    # write images out
    for frame in frames:
        videoOut.write(frame)
    videoOut.release()
    return output_path

def videoOut2Cloud(output_path, fileID):
    file_out = fileID + "out"
    cloudinary.uploader.upload(output_path, public_id=file_out, resource_type = "video", format="mp4", folder="samples/")
    return file_out

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