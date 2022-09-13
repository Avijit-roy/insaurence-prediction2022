from flask import Flask, render_template, request, jsonify
from pycaret.regression import *
import pandas as pd
import numpy as np

model = load_model('deployment_28042020')
model
import requests
url = 'https://pycaret-insurance.herokuapp.com/predict_api'
pred = requests.post(url,json={'age':19, 'sex':'male', 'bmi':24.6, 'children':1, 'smoker':'no', 'region':'southwest'})
print(pred.json())
app = Flask(__name__)
cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
@app.route('/')
def home():
  return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
  int_features = [x for x in request.form.values()]
  final = np.array(int_features)
  data_unseen = pd.DataFrame([final], columns = cols)
  prediction = predict_model(model, data=data_unseen, round = 0)
  prediction = int(prediction.Label[0])
  return render_template('home.html', pred='Expected Bill will be {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)