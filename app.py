from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # To enable CORS for cross-origin requests
import joblib
import numpy as np

# Load the saved model
model = joblib.load('forex_model.pkl')

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Route to serve the frontend HTML page (Optional, if you want an HTML interface)
@app.route('/')
def home():
    return render_template('index.html')  # This renders the frontend HTML page (index.html)

# Route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request (the data should be in JSON format)
        data = request.get_json(force=True)
        
        # Extract features from the data (assuming data contains 'features' as a list)
        features = np.array(data['features']).reshape(1, -1)  # Reshape for the model
        
        # Predict using the loaded model
        prediction = model.predict(features)
        
        # Map prediction to 'Buy' or 'Sell'
        result = "Buy" if prediction == 1 else "Sell"
        
        # Send back the prediction result
        return jsonify(prediction=result)
    except Exception as e:
        return jsonify(error=str(e)), 400

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)  # 0.0.0.0 for external access on local network, port 5000
