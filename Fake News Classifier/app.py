# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 23:42:35 2023

@author: ayush
"""
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template


# Making an app
app= Flask(__name__)

#Loading the Pickle model
model= pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    string_features=[(x) for x in request.form.values()]
    features= [np.array(string_features)]
    prediction= model.predict(features)
    
    return render_template("index.html", prediction_text= "HELLO {}".format(prediction)) 

if __name__ == "__main__":
    app.run(debug=True)