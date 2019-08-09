from celery import Celery
import testModel2 as tm2
import json
from keras.models import model_from_json

app = Celery()
app.config_from_object("celery_settings")

# Model Pipeline Defined
@app.task(bind=True)
def evaluateSquat(self,file_source, output_path, file_name, weights_path):    
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
    trained_model = tm2.loadTrainedModel(weights_path)
    predictions   = tm2.makepredictions(frames, trained_model)
    goodSquats    = predictions[predictions==0].shape[0]
    badSquats     = predictions[predictions==1].shape[0]

    # Label images based on predictions
    frames        = tm2.createLabeledImages(orig_frames,predictions)
    # Make output GIF of Labeled images
    movie         = tm2.videoOutput(frames, output_path)
    print("Predictions: ", predictions)
    return {'result' : file_name, 'status' : "SUCCESS"}
    