from celery import Celery
import testModel2 as tm2
import json
from keras.models import model_from_json

app = Celery()
app.config_from_object("celery_settings")

# Model Pipeline Defined
@app.task(bind=True)
def evaluateSquat(self,file_source, base_model, trained_model, output_path, file_name):    
    self.update_state(state='STARTED', meta={'status' : "STARTED"})
    # Make image frames from video
    frames      = tm2.makeFrames(file_source)
    orig_frames = frames

    # Preprocess frames
    frames = tm2.processImages(frames)
    # Classify images based on vgg16 base model
    base_model = model_from_json(base_model)
    frames     = tm2.classifyImages(frames, base_model)
    
    # Make predictions
    trained_model = model_from_json(trained_model)
    predictions   = tm2.makepredictions(frames, trained_model)
    goodSquats  = predictions[predictions==0].shape[0]
    badSquats   = predictions[predictions==1].shape[0]
    # Label images based on predictions
    frames        = tm2.createLabeledImages(orig_frames,predictions)
    # Make output GIF of Labeled images
    movie         = tm2.videoOutput(frames, output_path)
    print("I evaluated a squat!!!")
    return {'result' : file_name, 'status' : "SUCCESS"}
    