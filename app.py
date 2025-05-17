#import flask 
from flask import Flask, request, jsonify
#importing pickle
import pickle
#importing pandas
import pandas as pd
from PhishingPredictor import EmailPreProcessor  

#app flask
app = Flask(__name__)
#pickle loading file
model = pickle.load(open("final_model.pkl", "rb"))

#route app, methods: POST
@app.route("/prediction-check", methods=["POST"])
#function for prediction 
def predict_fishing():
  #data request json
    data = request.json
  #data for the email
    data_for_email = pd.DataFrame([data])
    prediction = model.predict(data_for_email)
    return jsonify({"prediction": int(prediction[0])})

#main
if __name__ == "__main__":
    app.run(debug=True)
