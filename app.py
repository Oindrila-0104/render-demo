from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)


# Load the saved crop prediction model
import pickle
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        nitrogen = float(request.form['Nitrogen'])
        phosphorus = float(request.form['Phosphorus'])
        potassium = float(request.form['Potassium'])
        temperature = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        # Combine features into a single array
        features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])

        # Initialize and fit StandardScaler for the current features
        scaler = StandardScaler()
        features = scaler.fit_transform(features)  # Fit and transform the features

        # Predict using the model
        prediction = model.predict(features)[0]

        return render_template('index.html', prediction=f"Recommended Crop: {prediction}")

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)