from flask import Flask, render_template, flash, request, redirect, session
from werkzeug import secure_filename
import testModel2 as tm2
import os

app = Flask(__name__)
uploads_dir = os.path.join(app.instance_path, 'uploads')
print(uploads_dir)
static_dir  = 'static/images'
os.makedirs(static_dir, exist_ok=True)
os.makedirs(uploads_dir,exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_file')
def upload_file():
    return render_template('/upload_file.html')

@app.route('/run_test', methods=['GET', 'POST'])
def run_test():
    if request.method == 'POST':
        file = request.files['file']
        file.save(os.path.join(uploads_dir, secure_filename(file.filename)))
        tm2.makeFrames(file.filename, uploads_dir)
        return redirect('/results')

@app.route('/results')
def results():
    # load model, test and show results
    test_data  = tm2.loadInTestImages(os.path.join(uploads_dir,"newImports.csv"))
    orig_image = test_data[3]
    test_image = test_data[2]
    test_y     = test_data[1]
    numTests   = test_data[0]

    test_image = tm2.load_basemodel(test_image, numTests)
    model      = tm2.loadTrainedModel('trained_model.h5')

    predictions = tm2.makepredictions(model, test_image)
    goodSquats  = predictions[predictions==0].shape[0]
    badSquats   = predictions[predictions==1].shape[0]
    labeledImgs = tm2.createLabeledImages(orig_image, predictions)
    movie       = tm2.videoOutput(labeledImgs,os.path.join(static_dir,'movie.gif'))

    return render_template('/results.html', goodSquats=goodSquats, badSquats=badSquats, movie=movie)

if __name__ == '__main__':
    app.run(debug=True)