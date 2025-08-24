# /backend/api.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# THIS IS PART OF THE CONNECTION
# Enable Cross-Origin Resource Sharing (CORS) for all routes.
# This allows your Next.js app (on port 3000) to make requests
# to this Flask app (on port 5000).
CORS(app)

# --- Load the Trained Model ---
try:
    # Make sure you are loading your modern, retrained model file
    model = joblib.load('fraud_detection_model_V2.pkl')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Model file not found. Ensure 'fraud_detection_model_V2.pkl' is in the backend directory.")
    model = None


# --- THE BACKEND CONNECTION POINT ---
# This decorator tells Flask to listen for POST requests at the '/predict' URL.
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the model was loaded correctly
    if model is None:
        return jsonify({'error': 'Model is not loaded, cannot make a prediction.'}), 500

    # 1. Receive the data (the "order") from the Next.js frontend
    json_data = request.get_json()
    if not json_data:
        return jsonify({'error': 'No input data provided.'}), 400

    # 2. Convert the data into the format the model expects (a pandas DataFrame)
    try:
        input_df = pd.DataFrame([json_data])
    except Exception as e:
        return jsonify({'error': f'Error processing input data: {str(e)}'}), 400

    # 3. Use the model to make a prediction
    try:
        prediction = model.predict(input_df)
        
        # Convert the prediction (0 or 1) into a human-readable string
        result = "FAKE" if prediction[0] == 1 else "REAL"
        
        # 4. Send the result back to the frontend (the "food")
        return jsonify({'prediction': result})
        
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

# --- Start the Server ---
if __name__ == '__main__':
    # The API will run on http://127.0.0.1:5000
    app.run(debug=True, port=5000)