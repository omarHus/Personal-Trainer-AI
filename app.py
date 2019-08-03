from flask import Flask, render_template, request, redirect, jsonify, Response
from werkzeug import secure_filename
from keras.utils.data_utils import get_file
import testModel2 as tm2
import os
import json

app = Flask(__name__)
uploads_dir = os.path.join(app.instance_path, 'uploads')
static_dir  = 'static/images'
os.makedirs(static_dir, exist_ok=True)
os.makedirs(uploads_dir,exist_ok=True)

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

# When file is uploaded this method runs, loads the trained model and makes predictions
newFrames = None
movie = None
goodSquats = None
badSquats = None
resp = None
@app.route('/run_test', methods=['GET', 'POST'])
def run_test():
    #Handle user video file input
    if request.method == 'POST':
        # Get json text showing that file has been uploaded directly to cloudinary successfully
        response = request.form['javascript_data']
        response =  json.loads(response)
        #Make the images and test them against the model
        if response['secure_url']:
            global newFrames
            newFrames  = tm2.makeFrames(response['url']) #make image frames for predictions
            if newFrames is not None:
                data = {
                    'goodSquats' : "goodSquats",
                    'badSquats'  : "badSquats",
                    'movie'      : "movie"
                }
                global resp
                resp = jsonify(data)
                resp.status_code = 200
                return redirect('/load_model')
    return render_template('/error.html')

@app.route('/load_model')
def load_model():
    global test_data, orig_image, test_image, model
    test_data  = tm2.processImages(newFrames)
    orig_image = test_data[3]
    test_image = test_data[2]
    test_y     = test_data[1]
    numTests   = test_data[0]

    test_image = tm2.load_basemodel(test_image, numTests)
    weights_path = get_file('trained_model.h5','https://github.com/omarHus/physioWebApp/raw/master/trained_model.h5')
    model      = tm2.loadTrainedModel(weights_path)

    return redirect('/test_model')

@app.route('/test_model')
def test_model():
    global goodSquats, badSquats, movie
    predictions = tm2.makepredictions(model, test_image)
    goodSquats  = predictions[predictions==0].shape[0]
    badSquats   = predictions[predictions==1].shape[0]
    labeledImgs = tm2.createLabeledImages(orig_image, predictions)
    movie       = tm2.videoOutput(labeledImgs,os.path.join(static_dir,'movie.gif'))
    return resp

@app.route('/show_results')
def show_results():
    return render_template('results.html', goodSquats=goodSquats, badSquats=badSquats, movie=movie)

if __name__ == '__main__':
    app.run(debug=True)