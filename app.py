from flask import Flask;
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model;
from keras.preprocessing import image
import numpy as np
from flask import render_template, request;

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('INDEX.html')

@app.route('/get_doctor')
def get_doctor():
    return render_template('doctor.html')

@app.route('/skin_cancer')
def skin_cancer():
    return render_template('skin_cancer.html')

@app.route('/skin_cancer_pred',methods =["GET", "POST"])
def skin_cancer_pred():
    if request.method == "POST":
       file_name = request.form.get("file")
       img_path='skin/' + file_name
       print(img_path)
       class_names = ["Benign","Malignant"]
       reconstructed_model = keras.models.load_model("skin_cancer")
       new_img = image.load_img(img_path, target_size=(64, 64))
       img = image.img_to_array(new_img)
       img = np.expand_dims(img, axis=0)
       result = reconstructed_model.predict(img)
       prediction = np.argmax(result,axis=1)
       print(prediction)
       print(class_names[prediction[0]])
       print('inside skin_cancer_pred')
       return render_template('skin_cancer.html', result = class_names[prediction[0]])

@app.route('/brain_cancer')
def brain_cancer():
    return render_template('brain_cancer.html')

@app.route('/brain_cancer_pred',methods =["GET", "POST"])
def brain_cancer_pred():
    if request.method == "POST":
       file_name = request.form.get("file")
       img_path='brain/' + file_name
       print(img_path)
       class_names = ["Absent","Present"]
       reconstructed_model = keras.models.load_model("brain_cancer")
       new_img = image.load_img(img_path, target_size=(64, 64))
       img = image.img_to_array(new_img)
       img = np.expand_dims(img, axis=0)
       result = reconstructed_model.predict(img)
       prediction = np.argmax(result,axis=1)
       try:
           assert prediction[0] == None,"Assertion Passed, Continuing the process ... " 
       except:
           print('Assertion passed')
           print(prediction)
           print(class_names[prediction[0]])
           print('inside brain_cancer_pred')
           return render_template('brain_cancer.html', result = class_names[prediction[0]])
       

@app.route('/breast_cancer')
def breast_cancer():
    return render_template('breast_cancer.html')

@app.route('/breast_cancer_pred',methods =["GET", "POST"])
def breast_cancer_pred():
    if request.method == "POST":
       file_name = request.form.get("file")
       img_path='breast/' + file_name
       print(img_path)
       class_names = ["Benign","Malignant"]
       reconstructed_model = keras.models.load_model("breast_cancer")
       new_img = image.load_img(img_path, target_size=(244,244))
       img = image.img_to_array(new_img)
       img = np.expand_dims(img, axis=0)
       result = reconstructed_model.predict(img)
       prediction = np.argmax(result,axis=1)
       print(prediction)
       print(class_names[prediction[0]])
       print('inside breast_cancer_pred')
       return render_template('breast_cancer.html', result = class_names[prediction[0]])
    
if __name__ == '__main__':
    app.run(host='localhost', port='81',debug=True)




