import cv2
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import pandas as pd
from keras.preprocessing import image
import numpy as np
from keras.utils import np_utils
from skimage.transform import resize
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout
import os, ssl

##Import Mapping File And Create Training Set
# 0 = Good Squat
# 1 = Bad  Squat

data = pd.read_csv('Training/test.csv')
data.head()

X = [ ]
for img_name in data.Image_ID:
    img = plt.imread('Training/' + img_name)
    X.append(img)
X = np.array(X)

#Create Class Array
y       = data.Class
dummy_y = np_utils.to_categorical(y)

#Preprocess Input Data
image = []
for i in range(0,X.shape[0]):
    a = resize(X[i], output_shape=(224,224), preserve_range=True).astype(int)
    image.append(a)
X = np.array(image)
X = preprocess_input(X, mode='tf')

#Split data into training and Testing set
X_train, X_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.3, random_state=42)
trainingSetLength   = len(X_train)
validationSetLength = len(X_valid)

##Build Model
# If statement needed to get passed SSL bloc of python
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and 
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
X_train = base_model.predict(X_train)
X_valid = base_model.predict(X_valid)
X_train.shape, X_valid.shape

X_train = X_train.reshape(trainingSetLength, 7*7*512)
X_valid = X_valid.reshape(validationSetLength, 7*7*512)
train   = X_train/X_train.max()
X_valid = X_valid/X_train.max()

# i. Building the model
model = Sequential()
model.add(InputLayer((7*7*512,)))    # input layer
model.add(Dense(units=1024, activation='sigmoid')) # hidden layer
model.add(Dense(2, activation='softmax'))    # output layer
model.summary()

# ii. Compiling the model (loss='categorical_crossentropy')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# iii. Training the model
model.fit(train, y_train, epochs=30, validation_data=(X_valid, y_valid))

# Save model to file to be used in other predictions
model.save_weights('trained_model.h5')

print("Model Trained!")