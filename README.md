# Personal Trainer AI App

## Goal:
- Personal Trainer in your pocket app.
- Provide user live feedback on their movements during specific exercises.
- Use machine learning models to classify correct and incorrect movement.

## Pre-requisites
- Git
- Python 3.7.x
- pip
- virtualenv
- activate virtualenv (myenv) with command line command: source myenv/bin/activate
- see requirements.txt file to set up virtual env (pip3 install -r requirements.txt)
- For cloudinary: make sure to run these config vars commands in terminal before starting (i'll email you the api_secret key):
        export CLOUDINARY_CLOUD_NAME="aitrainer"  
        export CLOUDINARY_API_KEY="862358378255759"  
        export CLOUDINARY_API_SECRET="****************" 

## How to Run on Local Host
- Go to local directory. From command line: python3 app.py
- Copy the url: http://127.0.0.1:5000/ and paste in a browser
- or command line: flask run

## Development
- app.py:            Flask app deployed on Heroku.
- testModel2.py:     All methods used for main app are held here.
- trainModel_module: Used to train model. Trained model is held in public github repo: https://github.com/omarHus/physioWebApp/blob/master/trained_model.h5
- vid2Frame_func.py  Methods for creating images from video files.

## Heroku
- get heroku cli -> google heroku cli and follow the steps for either mac or windows
- App is served at https://squatapp.herokuapp.com/
- Provide me your email and I can add you as a collaborator on Heroku
- from your local git repo you can push to heroku with: git push heroku master
- Max 500Mb push to heroku
- to open heroku browser make sure the server is on with command line: heroku ps. Should say there is a web dyno active
- if not, command line: heroku ps:scale web=1
- This will open server and host site.
- then open site with command line: heroku open

## To do:
- link in results.html to revert to homepage to try again.
- Heroku Deployment Error: R14 - Memory quoate exceeded
    - The web dyno (server process) on heroku gets overwhelmed by the file downloads we make.
    - Start a worker dyno
- Create postgres sql database to keep track of filenames and paths (integrate with Heroku)
- Server side (flask app.py): authenticate clients and keep track of sessions so that we don't have people posting same video multiple times and one user trying to upload too many videos