from joblib import load
from flask import Flask, render_template, request
import pandas as pd

# Load trained model and label encoder
model = load('Crop_prediction_model.joblib')
label_encoder = load('Crop_label_encoder.joblib')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get inputs
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            water_availability = float(request.form['water availability'])


            # Create input DataFrame
            input_data = pd.DataFrame([[
                temperature, humidity, ph, water_availability
            ]], columns=['temperature', 'humidity', 'ph', 'water availability'])

            # Predict
            prediction_encoded = model.predict(input_data)[0]

            # Decode to crop name
            predicted_crop = label_encoder.inverse_transform([prediction_encoded])[0]

            return render_template('index.html',
                                   prediction_text=f"Recommended Crop: {predicted_crop}")

        except Exception as e:
            return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
