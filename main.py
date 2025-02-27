import joblib
import numpy as np
from flask import Flask,render_template,request
#initialize flask app
app=Flask(__name__)
#load the model and scaler pickle files
model=joblib.load("model_scaled.pkl")
scaler=joblib.load("scaled.pkl")
#logic
@app.route("/") #decorator
def home():
    return render_template("home.html") 
@app.route("/submit",methods=['post']) #decorator
def submit():
    data=[eval(data) for data in request.form.values()]
    data_array=np.array([data])
    scaled_input=scaler.transform(data_array)
    result=model.predict(scaled_input)
    if result[0]==1:
        return " diabetic"
    return "non diabetic"
#run the application
app.run(host="0.0.0.0",port=7080,debug=True)
