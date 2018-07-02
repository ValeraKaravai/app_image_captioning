import json
import os

from flask import Flask, render_template, request, send_from_directory
from model.main import apply_model_to_image_raw_bytes, save_pic_to_image_raw_bytes
from model.utils import download_file
import sys
import uuid


UPLOAD_FOLDER = 'data/image/original'
UPLOAD_FOLDER_CROP = 'data/image/crop'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_FOLDER_CROP'] = UPLOAD_FOLDER_CROP


@app.route("/", methods=["POST", "GET", 'OPTIONS'])
def index_page():

    return render_template('index.html')


@app.route('/hook2', methods=["POST", "GET", 'OPTIONS'])
def predict():
    if request.method == 'POST':
        filename = request.values['filename']
        prediction = apply_model_to_image_raw_bytes(open((app.config['UPLOAD_FOLDER'] + '/{}').format(filename), "rb").read(),
                                                    filename=filename)
        return prediction
    return 'not'


@app.route('/upload', methods=['POST'])
def upload_file():
    unique_filename = str(uuid.uuid4().hex) + '.png'
    f = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

    if request.form['ref'] != '':

        download_file(request.form['ref'], f)
    else:
        file = request.files['image']
        file.save(f)

    return unique_filename


@app.route('/<filename>')
def uploaded_file_2(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER_CROP'],
                               filename)


@app.route('/hook3', methods=['POST'])
def uploaded_file():
    filename = request.values['filename']
    return filename


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
