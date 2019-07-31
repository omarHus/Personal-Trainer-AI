from flask import Flask, render_template, flash, request, redirect, session
from werkzeug import secure_filename
from keras.utils.data_utils import get_file
import testModel2 as tm2
import os

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
@app.route('/run_test', methods=['GET', 'POST'])
def run_test():
    #This is a server-side upload of the video file from the user -> would like this to be browser side direct to cloudinary
    if request.method == 'POST':
        file = request.files['file']
        response  = upload(file, folder="squat_videos", resource_type = "video") #cloud upload
        print("Response = ", response['secure_url'])
        if response:
            newFrames = tm2.makeFrames(response['secure_url']) #make image frames for predictions
            test_data  = tm2.processImages(newFrames)
            orig_image = test_data[3]
            test_image = test_data[2]
            test_y     = test_data[1]
            numTests   = test_data[0]

            test_image = tm2.load_basemodel(test_image, numTests)
            weights_path = get_file('trained_model.h5','https://github.com/omarHus/physioWebApp/raw/master/trained_model.h5')
            model      = tm2.loadTrainedModel(weights_path)

            predictions = tm2.makepredictions(model, test_image)
            goodSquats  = predictions[predictions==0].shape[0]
            badSquats   = predictions[predictions==1].shape[0]
            labeledImgs = tm2.createLabeledImages(orig_image, predictions)
            movie       = tm2.videoOutput(labeledImgs,os.path.join(static_dir,'movie.gif'))
            return render_template('/results.html', goodSquats=goodSquats, badSquats=badSquats, movie=movie) #sending data to html page to display
        else:
            print("ERROR: Could not upload from cloud.")
    return render_template('/upload_file.html')

if __name__ == '__main__':
    app.run(debug=True)