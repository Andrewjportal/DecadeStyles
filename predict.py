import base64
import io


import cv2
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array

from flask import request
from flask import Flask, render_template
from flask import jsonify
from flask_cors import CORS
import tensorflow as tf
keras=tf.keras



app = Flask(__name__)
CORS(app)

@app.route('/DecadeStyles')
def index():


    return render_template('DecadeStyles.html')

CATEGORIES = ['1920-1930s', '1940-1950s', '1960-1970s','1980-1990s','2000-2010']


def get_model():
    global model
    model = keras.models.load_model('hook_vgg16_best.model')
    print('* model loaded')

def prepare(image, target_size):
    if image.mode != 'RGB':
        image = image.conver('RGB')
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image


get_model()


model._make_predict_function()




@app.route('/predict', methods=['POST'])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    prediction = model.predict([prepare(image, target_size=(244, 244))])
    #predict outputs an array within array. [0] takes first array in array and np.argmax gives the position.
    response = (CATEGORIES[(np.argmax(prediction[0]))])
    return jsonify(response)
