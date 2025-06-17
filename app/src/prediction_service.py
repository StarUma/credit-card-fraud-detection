import pickle
import numpy as np

# Load the saved model (correct path)
model = pickle.load(open('models/fraud_detection_model.pkl', 'rb'))

def make_prediction(features: list):
    input_array = np.array([features])  # Shape: (1, 30)
    prediction = model.predict(input_array)
    return "Fraud" if prediction[0] == 1 else "Normal"
