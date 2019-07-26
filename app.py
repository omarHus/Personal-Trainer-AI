from flask import Flask, render_template, flash, request, redirect, session
from werkzeug import secure_filename

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, World!"

if __name__ == '__main__':
    app.run(debug=True)