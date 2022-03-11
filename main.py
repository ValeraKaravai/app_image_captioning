import os
from flask import Flask, render_template, request, send_from_directory, after_this_request
from model.main import Model
from model.utils import download_file
import uuid


UPLOAD_FOLDER = 'data/image/original'
UPLOAD_FOLDER_CROP = 'data/image/crop'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_FOLDER_CROP'] = UPLOAD_FOLDER_CROP
model = Model()


@app.route("/", methods=["POST", "GET", 'OPTIONS'])
def index_page():

    return render_template('index.html')


@app.route('/hook2', methods=["POST", "GET", 'OPTIONS'])
def predict():
    if request.method == 'POST':
        filename = request.values['filename']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'],request.values['filename'])
        prediction = model.apply_model_to_image_raw_bytes(open(filepath, "rb").read(),
                                                          filename=filename,
                                                          dir_save=app.config['UPLOAD_FOLDER_CROP'])
        save = model.save_pic_amazon(filename=filepath)
        # os.remove(filepath)
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
    file_path = os.path.join(app.config['UPLOAD_FOLDER_CROP'],
                             filename)
    file_handle = open(file_path, 'r')

    file_path_original = os.path.join(app.config['UPLOAD_FOLDER'],
                                      filename)
    file_handle_original = open(file_path_original, 'r')

    @after_this_request
    def remove_file(response):
        try:
            os.remove(file_path)
            file_handle.close()
            os.remove(file_path_original)
            file_handle_original.close()
        except Exception as error:
            app.logger.error("Error removing or closing downloaded file handle", error)
        return response
    return send_from_directory(app.config['UPLOAD_FOLDER_CROP'],
                               filename)


@app.route('/hook3', methods=['POST'])
def uploaded_file():
    filename = request.values['filename']
    return filename


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)


def pp():
    return