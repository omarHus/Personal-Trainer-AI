# Personal Trainer AI App

## Pre-requisites
- Git
- Python 3.7.x
- pip
- virtualenv
- activate virtualenv (myenv) with command line command: source myenv/bin/activate
- see requirements.txt file to set up virtual env (pip3 install -r requirements.txt)

## How to Run on Local Host
- Go to local directory. From command line: python3 app.py
- Copy the url: http://127.0.0.1:5000/ and paste in a browser

## Development
- app.py:            Flask app deployed on Heroku.
- testModel2.py:     All methods used for main app are held here.
- trainModel_module: Used to train model. Trained model is held in public github repo: https://github.com/omarHus/physioWebApp/blob/master/trained_model.h5
- vid2Frame_func.py  Methods for creating images from video files.