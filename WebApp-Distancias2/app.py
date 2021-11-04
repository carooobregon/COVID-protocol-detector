import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/upload'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10240 * 10000
app.config['UPLOAD_PATH'] = 'uploads'