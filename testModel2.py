def main():
    test_data  = loadInTestImages('instance/uploads/newImports.csv')
    test_image = test_data[2]
    test_y     = test_data[1]
    numTests   = test_data[0]
    orig_image = test_data[3]

    test_image = load_basemodel(test_image, numTests)
    model      = loadTrainedModel('trained_model.h5')

    predictions = makepredictions(model, test_image)
    new_images  = createLabeledImages(orig_image, predictions)
    videoOutput(new_images)
    # for img in new_images:
    #     cv2.imshow('new_img',img)
    #     cv2.waitKey(1000)
    # cv2.destroyAllWindows()
    # scores = model.evaluate(test_image, test_y)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

def classMap(classNumber):
    if classNumber == 0:
        return "Good Squat"
    else:
        return "Bad Squat"

def videoOutput(frames, output_path):
    imageio.mimsave(output_path, frames, duration=0.3)
    return output_path

def createLabeledImages(orig_images, labels):
    font  = cv2.FONT_HERSHEY_SIMPLEX
    count = 0
    for img in orig_images:
        if labels[count] == 0:
            color = (0,255,0)
        else:
            color = (255,0,0)
        cv2.putText(img, classMap(labels[count]), (0,int(244/2)), font, 2, color)
        count += 1
    cv2.destroyAllWindows()
    
    return orig_images

# Load in Test Images and PreProcess
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


# extracting features from the images using pretrained model
def load_basemodel(testImages, num_of_tests):
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