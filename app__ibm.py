import pandas as pd
from flask import Flask, request, jsonify, render_template
import json
import requests


# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "HSH_DnKXsrObyBjFChwLYk3pLkmNwlBnM2cPrXsCLV9b"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

print("mltoken",mltoken)

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def helloworld():
    return render_template("index.html")


@app.route('/input', methods=['GET','post'])
def hello():
    return render_template("input.html")


@app.route('/output', methods=['GET','post'])
def predict():
	
	GRE_Score = int(request.form['GRE Score'])
	TOEFL_Score = int(request.form['TOEFL Score'])
	University_Rating = int(request.form['University Rating'])
	SOP = float(request.form['SOP'])
	LOR = float(request.form['LOR'])
	CGPA = float(request.form['CGPA'])
	Research = int(request.form['Research'])
	payload_scoring = {"input_data": [{"fields": [["GRE Score","TOEFL Score","University Rating","SOP","LOR","CGPA","Research"]], "values": [[GRE_Score,TOEFL_Score,University_Rating,SOP,LOR,CGPA,Research]]}]}

	response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/20de2495-5ae1-4f85-ba14-000badf504be/predictions?version=2022-11-11', json=payload_scoring,
	headers={'Authorization': 'Bearer ' + mltoken})
	print("Scoring response")
	predictions=response_scoring.json()
	print(predictions)
	pred = predictions['predictions'][0]['values'][0][0]
	output=pred*100
	if(output>50):
		return render_template('output.html', Admission_Prediction='High')
	else:
		return render_template('output.html', Admission_Prediction='Low ')
		
	
	# return render_template('output.html', Admission_Prediction=output)
	
if __name__ == "__main__":
	app.run(debug=True)