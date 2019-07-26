from flask import Flask, render_template, flash, request, redirect, session
from werkzeug import secure_filename

app = Flask(__name__)

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
        print(file.filename)
        makeFrames(file.filename, uploads_dir)
        return redirect('/results')

if __name__ == '__main__':
    app.run(debug=True)