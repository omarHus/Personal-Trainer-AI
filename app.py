from flask import Flask, render_template, request, redirect, jsonify, Response, send_from_directory, url_for
from celery.result import AsyncResult
from celery import Celery
from keras.models import model_from_json
from keras.utils.data_utils import get_file
import testModel2 as tm2
import numpy as np
import json
import os

#####################################################
############### Flask App Setup #####################
app         = Flask(__name__)
uploads_dir = os.path.join(app.instance_path, 'uploads')
os.makedirs(uploads_dir, exist_ok=True)
app.config['uploads_dir'] = uploads_dir

#####################################################
################ Celery Setup #######################
celery = Celery(app.name)
celery.config_from_object("celery_settings")

# Model Pipeline Defined
@celery.task(bind=True)
def evaluateSquat(self,file_source, output_path, file_name, fileID):    
    self.update_state(state='STARTED', meta={'status' : "STARTED"})
    # Make image frames from video
    frames      = tm2.makeFrames(file_source)
    orig_frames = frames

    # Preprocess frames
    frames = tm2.processImages(frames)
    # Classify images based on vgg16 base model
    base_model = tm2.load_basemodel()
    frames     = tm2.classifyImages(frames, base_model)
    
    # Make predictions
    weights_path  = get_file('trained_model.h5','https://github.com/omarHus/physioWebApp/raw/master/trained_model.h5')
    trained_model = tm2.loadTrainedModel(weights_path)
    predictions   = tm2.makepredictions(frames, trained_model)
    goodSquats    = predictions[predictions==0].shape[0]
    badSquats     = predictions[predictions==1].shape[0]

    # Label images based on predictions
    frames        = tm2.createLabeledImages(orig_frames,predictions)
    # Make output GIF of Labeled images
    movie         = tm2.videoOutput(frames, output_path)
    movieID       = tm2.videoOut2Cloud(output_path, fileID)
    
    # remove file from memory and get rid of other memory leaks
    del trained_model, base_model, frames, orig_frames, movie, goodSquats, badSquats, predictions, weights_path

    for file in os.listdir(uploads_dir):
        file_path = os.path.join(uploads_dir,file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
    return {'result' : movieID, 'status' : "SUCCESS"}

#####################################################
############### Cloudinary Setup ####################
import cloudinary as cloud
from cloudinary.uploader import upload
from cloudinary.utils import cloudinary_url
import cloudinary.api
cloud.config(
  cloud_name = os.environ['CLOUDINARY_CLOUD_NAME'], 
  api_key    = os.environ['CLOUDINARY_API_KEY'],
  api_secret = os.environ['CLOUDINARY_API_SECRET'],
)

#####################################################
############### Flask Server Routing ################
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_file')
def upload_file():
    return render_template('/upload_file.html')

@app.route('/run_test', methods=['POST'])
def run_test():
    # Get json text showing that file has been uploaded directly to cloudinary successfully
    response    = request.get_json()
    fileSource  = response['url']
    fileID      = response['public_id']
    fileName    = fileID + ".mp4"
    output_path = os.path.join(uploads_dir,fileName)

    # Run main pipeline function using asynchronous worker dyno
    myPredictions = evaluateSquat.delay(fileSource, output_path, fileName, fileID)
    return jsonify({}), 202, {'Location': url_for('task_status', task_id=myPredictions.id)}

# This route is called on the client side by AJAX to poll when the squat has been evaluated
@app.route('/task_status/<task_id>')
def task_status(task_id):
    myTask = evaluateSquat.AsyncResult(task_id)
    if myTask.state == 'PENDING':
        # job did not start yet
        response = {
            'state': myTask.state,
        }
    elif myTask.state != 'FAILURE':
        response = {
            'state': myTask.state,
        }
        # Successfully evaluated squat. Return video to client side.
        if 'result' in myTask.info:
            movieID = "samples/" + myTask.info['result']
            response['result']  = cloudinary.CloudinaryVideo(movieID, format="mp4").video(width=300)
            # response['result'] = myTask.info['result']  
    else:
        # something went wrong in the background job
        response = {
            'state': myTask.state,
            'status': str(myTask.info),  # this is the exception raised
        }
    return jsonify(response)

# Get rid of memory leaks
@app.route('/reset')
def reset():
    for file in os.listdir(uploads_dir):
        file_path = os.path.join(uploads_dir,file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
    return redirect('/')

app.jinja_env.cache = {}
if __name__ == '__main__':
   app.run()