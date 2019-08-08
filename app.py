from flask import Flask, render_template, request, redirect, jsonify, Response, send_from_directory
from werkzeug import secure_filename
from keras.utils.data_utils import get_file
import testModel2 as tm2
import os
import numpy as np
import json
import keras.backend.tensorflow_backend
import tensorflow as tf
from keras.backend import clear_session

# Configuring Flask app and creating folder to hold gif for each user
app = Flask(__name__)
uploads_dir = os.path.join(app.instance_path, 'uploads')
os.makedirs(uploads_dir, exist_ok=True)
app.config['uploads_dir'] = uploads_dir

test_image = None
newFrames = None
orig_image = None
movie      = None
goodSquats = None
badSquats  = None
resp       = None

# Cloud server setup
import cloudinary as cloud
from cloudinary.uploader import upload
from cloudinary.utils import cloudinary_url
import cloudinary.api
cloud.config(
  cloud_name = os.environ['CLOUDINARY_CLOUD_NAME'], 
  api_key    = os.environ['CLOUDINARY_API_KEY'],
  api_secret = os.environ['CLOUDINARY_API_SECRET'],
) # This data is stored in config vars on heroku

#Homepage of website
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_file')
def upload_file():
    return render_template('/upload_file.html')

# When file is uploaded this method runs, 
# accepting file name thru post request 
# and creating frames from uploaded video
@app.route('/run_test', methods=['POST'])
def run_test():
    global movieName, resp, newFrames, orig_image, test_image, numTests,base_model, weights_path, model
    # Get json text showing that file has been uploaded directly to cloudinary successfully
    response = request.form['javascript_data']
    response =  json.loads(response)
    #Make the images and test them against the model
    if response['secure_url']:
        movieName = response['public_id'] + ".gif"
        print("Movie name is ", movieName)
        #make image frames for predictions
        newFrames = tm2.makeFrames(response['url'])
        try:
            newFrames
            numTests = len(newFrames)
            test_data  = tm2.processImages(newFrames)
            orig_image = test_data[3]
            test_image = test_data[2]
            numTests   = test_data[0]
            data = {
                'goodSquats' : "goodSquats",
                'badSquats'  : "badSquats",
                'movie'      : "movie"
            }
            # load model only once
            try:
                weights_path
            except:
                base_model   = tm2.load_basemodel()
                weights_path = get_file('trained_model.h5','https://github.com/omarHus/physioWebApp/raw/master/trained_model.h5')
                model        = tm2.loadTrainedModel(weights_path)
            
            #send json response back to javascript function
            resp = jsonify(data)
            resp.status_code = 200
            return resp
        except:
            data = {
                'status' : "error"
            }
            resp = jsonify(data)
            resp.status_code = 500
            return resp
    else:
        data = {
            'status' : "error"
        }
        resp = jsonify(data)
        resp.status_code = 500
        return resp

@app.route('/load_model', methods=['POST'])
def load_model():
    # Load the base model and preprocess the images
    global  base_model, numTests, test_image
    test_image = base_model.predict(test_image, batch_size=len(test_image))
    # # converting the images to 1-D form
    test_image = test_image.reshape(numTests, 7*7*512)
    # # zero centered images
    test_image = test_image/test_image.max()
    data = {
        "status" : "200 OK"
    }
    resp = jsonify(data)
    return resp

@app.route('/test_model', methods=['POST'])
def test_model():
    #Make predictions on the new images and make a gif
    global goodSquats, badSquats, movieName, movie, test_image, model, orig_image
    print("MovieName in test_model is ", movieName)
    predictions = tm2.makepredictions(model, test_image)
    goodSquats  = predictions[predictions==0].shape[0]
    badSquats   = predictions[predictions==1].shape[0]
    labeledImgs = tm2.createLabeledImages(orig_image, predictions)
    movie       = tm2.videoOutput(labeledImgs,os.path.join(uploads_dir,movieName))
    data = {
        "status" : "200 OK"
    }
    resp = jsonify(data)
    return resp

@app.route('/show_results')
def show_results():
    #Show the results of the predictions
    global movieName, goodSquats, badSquats
    filename = ""
    filename = movieName
    print("MovieName in show_results is ", movieName)
    return render_template('results.html', goodSquats=goodSquats, badSquats=badSquats, filename=filename)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(uploads_dir, filename)

@app.route('/reset')
def reset():
    global newFrames, movie, goodSquats, badSquats, resp, test_image, orig_image, test_data, movieName
    newFrames  = None
    movie      = None
    goodSquats = None
    badSquats  = None
    resp       = None
    test_image = None
    orig_image = None

    try:
        os.remove(os.path.join(uploads_dir, movieName))
    except:
        print("no file exists")

    return redirect('/')

app.jinja_env.cache = {}
if __name__ == '__main__':
   app.run()