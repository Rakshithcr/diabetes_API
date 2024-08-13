# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 09:12:51 2024

@author: v-rakshithcr
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn

# Define the model input schema using Pydantic
class ModelInput(BaseModel):
    pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Create the FastAPI app
app = FastAPI()

# Load the saved model
with open('diabetes_trained.sav', 'rb') as f:
    classifier = pickle.load(f)

# Index route
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# Route with a single parameter
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome ': f'{name}'}

# Expose the prediction functionality
@app.post('/predict')
def predict_diabetes(data: ModelInput):
    # Extract data from the request
    input_data = data.dict()
    
    input_list = [
        input_data['pregnancies'], input_data['Glucose'], input_data['BloodPressure'],
        input_data['SkinThickness'], input_data['Insulin'], input_data['BMI'],
        input_data['DiabetesPedigreeFunction'], input_data['Age']
    ]
    
    # Make prediction
    prediction = classifier.predict([input_list])
    
    # Return the prediction result
    if prediction[0] == 0:
        return {'prediction': 'The person is not diabetic'}
    else:
        return {'prediction': 'The person is diabetic'}

# Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
