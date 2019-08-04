from flask import Flask, render_template, request, redirect, jsonify, Response, send_from_directory
from werkzeug import secure_filename
from keras.utils.data_utils import get_file
import testModel2 as tm2
import os
import json

app = Flask(__name__)
uploads_dir = os.path.join(app.instance_path, 'uploads')
os.makedirs(uploads_dir, exist_ok=True)
app.config['uploads_dir'] = uploads_dir

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

newFrames = None
movie = None
goodSquats = None
badSquats = None
resp = None
weights_path = None
#Homepage of website
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_file')
def upload_file():
    return render_template('/upload_file.html')

# When file is uploaded this method runs, loads the trained model and makes predictions
@app.route('/run_test', methods=['GET', 'POST'])
def run_test():
    #Handle user video file input
    if request.method == 'POST':
        # Get json text showing that file has been uploaded directly to cloudinary successfully
        response = request.form['javascript_data']
        response =  json.loads(response)
        #Make the images and test them against the model
        if response['secure_url']:
            global newFrames, movieName
            movieName = response['public_id'] + ".gif"
            print("Movie name is ", movieName)
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
    global test_data, orig_image, test_image, model, weights_path, base_model
    if newFrames is not None:
        test_data  = tm2.processImages(newFrames)
        orig_image = test_data[3]
        test_image = test_data[2]
        test_y     = test_data[1]
        numTests   = test_data[0]

        if weights_path is None:
            base_model   = tm2.load_basemodel()
            weights_path = get_file('trained_model.h5','https://github.com/omarHus/physioWebApp/raw/master/trained_model.h5')
            model        = tm2.loadTrainedModel(weights_path)
            model._make_predict_function()

        test_image = base_model.predict(test_image)
        # converting the images to 1-D form
        test_image = test_image.reshape(numTests, 7*7*512)
        # zero centered images
        test_image = test_image/test_image.max()
    else:
        return resp
        return render_template('error.html')
    return redirect('/test_model')

@app.route('/test_model')
def test_model():
    global goodSquats, badSquats, movieName
    predictions = tm2.makepredictions(model, test_image)
    goodSquats  = predictions[predictions==0].shape[0]
    badSquats   = predictions[predictions==1].shape[0]
    labeledImgs = tm2.createLabeledImages(orig_image, predictions)
    movie       = tm2.videoOutput(labeledImgs,os.path.join(uploads_dir,movieName))
    return resp

@app.route('/show_results')
def show_results():
    return render_template('results.html', goodSquats=goodSquats, badSquats=badSquats, filename=movieName)

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
    

if __name__ == '__main__':
    app.run(debug=True)