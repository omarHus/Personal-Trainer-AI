from flask import Flask, render_template, request, redirect, jsonify, Response, send_from_directory, url_for
from celery.result import AsyncResult
from keras.utils.data_utils import get_file
import testModel2 as tm2
import os
import numpy as np
import json
from tasks import evaluateSquat

# Configuring Flask app and creating folder to hold gif for each user
app = Flask(__name__)
uploads_dir = os.path.join(app.instance_path, 'uploads')
os.makedirs(uploads_dir, exist_ok=True)
app.config['uploads_dir'] = uploads_dir

# load in models
base_model   = tm2.load_basemodel()
weights_path = get_file('trained_model.h5','https://github.com/omarHus/physioWebApp/raw/master/trained_model.h5')
model        = tm2.loadTrainedModel(weights_path)

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

@app.route('/run_test', methods=['POST'])
def run_test():
    global base_model, model
    # Get json text showing that file has been uploaded directly to cloudinary successfully
    response = request.get_json()
    #Make the images and test them against the model
    # try:
    fileSource  = response['url']
    fileID      = response['public_id']

    fileName    = fileID + ".gif"
    output_path = os.path.join('static/images',fileName)

    base_model_json = base_model.to_json()
    model_json      = model.to_json()
    myPredictions = evaluateSquat.delay(fileSource, base_model_json, model_json, output_path, fileName)
    return jsonify({}), 202, {'Location': url_for('task_status', task_id=myPredictions.id)}
    # except:
    #     print("Error uploading file")
    #     data = { "status" : "Error"}
    #     return jsonify(data)

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
        if 'result' in myTask.info:
            response['result'] = myTask.info['result']
            print("The result is ", response['result'])
    else:
        # something went wrong in the background job
        response = {
            'state': myTask.state,
            'status': str(myTask.info),  # this is the exception raised
        }
    return jsonify(response)

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