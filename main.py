import os
import random
import re
import warnings

import numpy as np
import skimage
import tensorflow as tf
from PIL import ImageDraw, ImageFont
from keras.backend import clear_session
from PIL import Image
from flask import Flask, send_file
from flask import request, jsonify
from flask_cors import CORS, cross_origin
from mrcnn.model import MaskRCNN, mold_image
from numpy import expand_dims

from config import PredictionConfig
from constant import SAVED_MODEL, TRAINED_MODEL, colors
import constant as const
from logger.logger import logger

import global_variable as gv

warnings.filterwarnings("ignore")

ALLOWED_EXTENSIONS = {'jpeg', 'jpg', 'png'}
tmp_upload_path = 'tmp/uploads/'
tmp_output_path = 'tmp/output/'

app = Flask(__name__)
cors=CORS(app)


def allowed_file(filename):
    """
        This Function checks whether file uploaded by User is allowed or not.

    :param filename: it is filename which has extension with it.
    :return: True/False (Whether file is allowed or not.)
    """

    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/detect', methods=['GET', 'POST'])
@cross_origin()
def detectEntity():
    """
        Main Route for Recognizing the objects in Photograph or Video.
    :return: It returns same file with recognized user.
    """
    if request.method == 'POST':
        if (request.files['file'].filename != "") and allowed_file(request.files['file'].filename):
            file = request.files['file']
            file_name = request.files['file'].filename
            file_path = os.path.join(tmp_upload_path, file_name)
            file.save(file_path)
            logger.info("File saved..!")
        else:
            logger.warn("In-appropriate File passed...!")
            return jsonify({"status": "failed", "error": "In-appropriate File passed...!"}), 500

        extn = file_name.rsplit('.')[1].lower()

        if extn in ['jpeg', 'jpg', 'png']:
            try:
                img = Image.open(file_path)
                img = np.array(img)
                img = skimage.color.gray2rgb(img)
                img = Image.fromarray(img.astype('uint8'), 'RGB')

                image = np.array(img)

                # convert pixel values (e.g. center)
                scaled_image = mold_image(image, gv.cfg)
                # convert image into one sample
                sample = expand_dims(scaled_image, 0)

                with graph.as_default():
                    yhat = gv.model.detect(sample, verbose=0)[0]

                print('# ' + '-' * 40 + '  Result  ' + '-' * 40 + ' #')
                print("yhat: ", yhat)
                print("Positions: ", yhat['rois'])
                print("Classes: ", yhat['class_ids'])
                print("Scores: ", yhat['scores'])
                print('# ' + '-' * 90 + ' #')

                for cls, box, score in zip(yhat['class_ids'], yhat['rois'], yhat['scores']):
                    classes = ['background'] + const.classes
                    predicted_class = classes[int(cls)]
                    label = '{} {:.2f}'.format(predicted_class, score)

                    y1, x1, y2, x2 = box
                    color = colors[cls]

                    draw = ImageDraw.Draw(img)
                    size = np.floor(1e-2 * img.size[1] + 0.5).astype('int32')
                    font = ImageFont.truetype(font='font/FiraMono-Medium.otf', size=size)
                    label_size = draw.textsize(label, font)

                    if y1 - label_size[1] >= 0:
                        text_origin = np.array([x1, y1 - label_size[1]])
                    else:
                        text_origin = np.array([x1, y1 + 1])

                    # create Box, Add Text
                    draw.rectangle([(x1, y2), (x2, y1)], outline=color, width=3)
                    draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=color)
                    draw.text(text_origin, label, fill=(255, 255, 255), font=font)
                    del draw

                out_file = os.path.join(tmp_output_path, file_name)
                img.save(out_file)

                return send_file(out_file, mimetype='image/jpg'), 200

            except Exception as e:
                logger.exception(e)
                return jsonify({"status": "failed", "error": "Error occurred while processing File...!"}), 500

    else:
        logger.warn("In-appropriate Request...!")
        return jsonify({"status": "failed", "error": "In-appropriate Request...!"}), 404


if __name__ == '__main__':
    graph = tf.get_default_graph()

    os.mkdir('logs/') if not os.path.isdir('logs/') else None
    os.makedirs(tmp_upload_path) if not os.path.isdir(tmp_upload_path) else None
    os.makedirs(tmp_output_path) if not os.path.isdir(tmp_output_path) else None

    # create config
    gv.cfg = PredictionConfig()
    # define the model
    gv.model = MaskRCNN(mode='inference', model_dir=SAVED_MODEL, config=gv.cfg)
    # load model weights
    print('# ' + '-' * 40 + '  Loading Model..!  ' + '-' * 40 + ' #')
    gv.model.load_weights(TRAINED_MODEL, by_name=True)
    print('# ' + '-' * 40 + '  Model Loaded Successfully..!  ' + '-' * 40 + ' #')

    app.run(debug=False, host='0.0.0.0', port=5000)
