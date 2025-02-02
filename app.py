from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Enable CORS
import joblib
import numpy as np

# Load the saved model
try:
    model = joblib.load('forex_model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Avoid crashing if model isn't found

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

@app.route('/')
def home():
    return jsonify({"message": "Forex Trading Signal API is Running!"})

@app.route('/api/get-trading-signals', methods=['GET'])
def get_trading_signals():
    # Simulate some trading signals (You can modify this with actual data)
    signals = [
        {"signal": "Buy EUR/USD", "price": 1.2100},
        {"signal": "Sell GBP/USD", "price": 1.3450},
        {"signal": "Buy USD/JPY", "price": 110.50}
    ]
    
    return jsonify(signals)  # Return the list of signals as JSON

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not found"}), 500  # Ensure model exists

    try:
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({"error": "Invalid request. Expected JSON with 'features'"}), 400

        # Convert JSON features into NumPy array
        features = np.array(data["features"]).reshape(1, -1)  

        # Make prediction
        prediction = model.predict(features)[0]  # Extract scalar value
        confidence = round(max(model.predict_proba(features)[0]), 2) if hasattr(model, "predict_proba") else None

        # Convert prediction to human-readable result
        result = "Buy" if prediction == 1 else "Sell"

        # Return JSON response
        return jsonify({
            "signal": result,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))  # Render provides a PORT dynamically
    app.run(debug=True, host='0.0.0.0', port=port)
