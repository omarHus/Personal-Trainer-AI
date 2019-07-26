from flask import Flask, render_template, flash, request, redirect, session
from werkzeug import secure_filename

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_file')
def upload_file():
    return render_template('/upload_file.html')

if __name__ == '__main__':
    app.run(debug=True)