import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
	return render_template('ibm.html')

@app.route('/predict', methods=['GET','post'])
def predict():
	GRE_Score = int(request.form['GRE Score'])
	TOEFL_Score = int(request.form['TOEFL Score'])
	University_Rating = int(request.form['University Rating'])
	SOP = float(request.form['SOP'])
	LOR = float(request.form['LOR'])
	CGPA = float(request.form['CGPA'])
	Research = int(request.form['Research'])
	
	final_features = pd.DataFrame([[GRE_Score, TOEFL_Score, University_Rating, SOP, LOR, CGPA, Research]])
	
	predict = model.predict(final_features)
	
	output = predict[0]
	
	out = round(output*100)

	out = 0 if out < 0 else out
	out = 100 if out > 100 else out

	return render_template('ibm.html', prediction_text='Admission chances are {} %'.format(out))
	
if __name__ == "__main__":
	app.run(debug=True)