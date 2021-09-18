from flask import Flask, flash, request, redirect, url_for, render_template, jsonify, send_from_directory
import urllib.request
import os
from werkzeug.utils import secure_filename

from predict import predict_image

app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html',)
 
@app.route('/uploadImage', methods=['POST', 'GET'])
def upload_image():
    if 'file' not in request.files:
        print('No file part')
        return redirect(request.url)
    file = request.files['file']

    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        prediction = predict_image(file_path)

        return jsonify(prediction)

    else:
        print('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
        
@app.route('/templates/<path:path>')
def send_static(path):
    return send_from_directory('templates', path, cache_timeout=0)

if __name__ == "__main__":
    app.run()

