from celery import Celery
import testModel2 as tm2
import json
from keras.models import model_from_json

app = Celery()
app.config_from_object("celery_settings")

# Model Pipeline Defined
@app.task
def evaluateSquat(filename, base_model, trained_model, output_path):    
    print("i'm in")
    # Make image frames from video
    frames      = tm2.makeFrames(filename)
    orig_frames = frames

    # Preprocess frames
    frames = tm2.processImages(frames)
    # Classify images based on vgg16 base model
    base_model = model_from_json(base_model)
    frames     = tm2.classifyImages(frames, base_model)
    
    # Make predictions
    trained_model = model_from_json(trained_model)
    predictions   = tm2.makepredictions(frames, trained_model)
    # Label images based on predictions
    frames        = tm2.createLabeledImages(orig_frames,predictions)
    # Make output GIF of Labeled images
    movie         = tm2.videoOutput(frames, output_path)

    print("I evaluated a squat!!!")
    return movie
    