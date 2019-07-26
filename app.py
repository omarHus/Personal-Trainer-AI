from flask import Flask, render_template, flash, request, redirect, session
from werkzeug import secure_filename
import testModel2
import os
import cv2
import math
import matplotlib.pyplot as plt
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
import ssl
import imageio
import ffmpeg
import os.path
from os import path

app = Flask(__name__)
uploads_dir = os.path.join(app.instance_path, 'uploads')
static_dir  = 'static/images'
os.makedirs(static_dir, exist_ok=True)
os.makedirs(uploads_dir,exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_file')
def upload_file():
    return render_template('/upload_file.html')

@app.route('/run_test', methods=['GET', 'POST'])
def run_test():
    if request.method == 'POST':
        file = request.files['file']
        file.save(os.path.join(uploads_dir, secure_filename(file.filename)))
        tm.makeFrames(file.filename, uploads_dir)
        return redirect('/results')

@app.route('/results')
def results():
    # load model, test and show results
    test_data  = tm.loadInTestImages(os.path.join(uploads_dir,"newImports.csv"))
    orig_image = test_data[3]
    test_image = test_data[2]
    test_y     = test_data[1]
    numTests   = test_data[0]

    test_image = tm.load_basemodel(test_image, numTests)
    model      = tm.loadTrainedModel('https://github.com/omarHus/physioWebApp/blob/master/trained_model.h5')

    predictions = tm.makepredictions(model, test_image)
    goodSquats  = predictions[predictions==0].shape[0]
    badSquats   = predictions[predictions==1].shape[0]
    labeledImgs = tm.createLabeledImages(orig_image, predictions)
    movie       = tm.videoOutput(labeledImgs,os.path.join(static_dir,'movie.gif'))

    return render_template('/results.html', goodSquats=goodSquats, badSquats=badSquats, movie=movie)

if __name__ == '__main__':
    app.run(debug=True)