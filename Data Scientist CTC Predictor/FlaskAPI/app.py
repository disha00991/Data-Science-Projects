import flask
from flask import Flask, render_template, request
import json
from model_prediction import predict_salary
import numpy as np
from pickle import load
from icecream import ic

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features= {
        'rating': float(request.form['rating']),
        'founded': int(request.form['founded']),
        'competitors': int(request.form['competitors']),
        'sector': request.form['sector'],
        'ownership': request.form['ownership'],
        'job_title': request.form['job_title'],
        'job_in_headquarters': 1 if request.form['job_in_headquarters'] == 'yes' else 0,
        'job_seniority': request.form['job_seniority'],
        'job_skills': request.form['job_skills']
        }
        model = load(open('model/data_scientist_salary_prediction_model.pkl', 'rb'))
        rating_scaler = load(open('data prep objects/rating_scaler.pkl', 'rb'))
        company_founded_scaler = load(open('data prep objects/company_founded_scaler.pkl', 'rb'))
        prediction = predict_salary(model, rating_scaler, company_founded_scaler, features)
        
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)