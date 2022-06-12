import os
import pathlib
from flask import Flask, url_for, redirect,  render_template, request
from tensorflow.keras.preprocessing import image
from cv2 import cv2 as cv2 
import numpy as np
import keras
import tensorflow as tf
from keras.models import load_model
import pandas as p


# 取得目前檔案所在的資料夾 
SRC_PATH =  pathlib.Path(__file__).parent.absolute()
UPLOAD_FOLDER = os.path.join(SRC_PATH,  'static', 'uploads')
model = load_model('m1.h5')

app = Flask(__name__)
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    file = request.files['filename']
    if file.filename != '':
        file.save(os.path.join(UPLOAD_FOLDER, file.filename))
    #數字    
    #img = image.load_img(file)
    def predict_prob(number):
        return [number[0],1-number[0]]
    file_name='./src/static/uploads/'+file.filename
    file_name2='./static/uploads/'+file.filename
    img_width, img_height = 300, 300
    img = image.load_img(file_name, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = x/255.
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    model1 = load_model('./model/model_1.h5')
    model2 = load_model('./model/model_2.h5')
    model3 = load_model('./model/model_3.h5')
    model4 = load_model('./model/model_4.h5')
    model5 = load_model('./model/model_5.h5')
    model6 = load_model('./model/model_6.h5')
    model7 = load_model('./model/model_7.h5')
    model8 = load_model('./model/model_8.h5')
    model9 = load_model('./model/model_9.h5')
    model10 = load_model('./model/model_10.h5')
    classes1 = np.array(list(map(predict_prob, model1.predict(images))))
    classes2 = np.array(list(map(predict_prob, model2.predict(images))))
    classes3 = np.array(list(map(predict_prob, model3.predict(images))))
    classes4 = np.array(list(map(predict_prob, model4.predict(images))))
    classes5 = np.array(list(map(predict_prob, model5.predict(images))))
    classes6 = np.array(list(map(predict_prob, model6.predict(images))))
    classes7 = np.array(list(map(predict_prob, model7.predict(images))))
    classes8 = np.array(list(map(predict_prob, model8.predict(images))))
    classes9 = np.array(list(map(predict_prob, model9.predict(images))))
    classes10 = np.array(list(map(predict_prob, model10.predict(images))))
    a1=classes1[0][0]
    b1=classes1[0][1]
    a2=classes2[0][0]
    b2=classes2[0][1]
    a3=classes3[0][0]
    b3=classes3[0][1]
    a4=classes4[0][0]
    b4=classes4[0][1]
    a5=classes5[0][0]
    b5=classes5[0][1]
    a6=classes6[0][0]
    b6=classes6[0][1]
    a7=classes7[0][0]
    b7=classes7[0][1]
    a8=classes8[0][0]
    b8=classes8[0][1]
    a9=classes9[0][0]
    b9=classes9[0][1]
    a10=classes10[0][0]
    b10=classes10[0][1]
 
    return render_template('final.html', \
    real1 = a1, fake1 = b1, \
    real2 = a2, fake2 = b2, \
    real3 = a3, fake3 = b3, \
    real4 = a4, fake4 = b4, \
    real5 = a5, fake5 = b5, \
    real6 = a6, fake6 = b6, \
    real7 = a7, fake7 = b7, \
    real8 = a8, fake8 = b8, \
    real9 = a9, fake9 = b9, \
    real10 = a10, fake10 = b10, file_name=file_name2)
    #return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=3000, debug=True)


