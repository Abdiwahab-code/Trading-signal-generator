from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import time
import os
import gdown
from twelvedata import TDClient

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Google Drive File ID
FILE_ID = "1x-ZeuGxIFn_3ozqe98wp7bLoKC7h_S6S"
MODEL_PATH = "new_forex_model_twelvedata.pkl"

# Download the model if not already present
if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    print("Downloading model from Google Drive...")
    gdown.download(url, MODEL_PATH, quiet=False)

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Avoid crashing if model isn't found

# Set up Twelve Data API client
API_KEY = "83d6a28890304e40b220b588e5e8359e"
td = TDClient(apikey=API_KEY)

# Define currency pairs and timeframes
currency_pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD"]
timeframes = ["5min", "15min", "1h"]

# Function to fetch live forex data
def fetch_live_forex_data():
    forex_data = {}
    for pair in currency_pairs:
        forex_data[pair] = {}
        for tf in timeframes:
            try:
                data = td.time_series(symbol=pair, interval=tf, outputsize=1, timezone="UTC").as_pandas()
                if not data.empty:
                    latest = data.iloc[-1]
                    forex_data[pair][tf] = {
                        "close": round(latest["close"], 5),
                        "high": round(latest["high"], 5),
                        "low": round(latest["low"], 5)
                    }
            except Exception as e:
                print(f"Error fetching {pair} at {tf}: {e}")
                forex_data[pair][tf] = None  # Handle errors gracefully
            time.sleep(1)  # Prevent API rate limit issues
    return forex_data

@app.route('/')
def home():
    return jsonify({"message": "Forex Trading Signal API is Running!"})

@app.route('/api/predict', methods=['GET'])
def get_trading_signals():
    # Fetch live forex data
    live_prices = fetch_live_forex_data()
    signals = []
    
    for pair, time_frames in live_prices.items():
        for tf, data in time_frames.items():
            if data is None:
                continue  # Skip if no data

            # Prepare feature set for prediction
            features = np.array([data["close"], data["high"], data["low"]]).reshape(1, -1)
            
            if model:
                prediction = model.predict(features)[0]
                confidence = round(max(model.predict_proba(features)[0]), 2) if hasattr(model, "predict_proba") else None
            else:
                prediction, confidence = "Unknown", None
            
            # Interpret prediction
            signal = "Buy" if prediction == 1 else "Sell"
            
            # Calculate Stop Loss & Take Profit
            stop_loss = round(data["close"] * 0.995, 5) if signal == "Buy" else round(data["close"] * 1.005, 5)
            take_profit = round(data["close"] * 1.005, 5) if signal == "Buy" else round(data["close"] * 0.995, 5)
            
            # Append result
            signals.append({
                "currency_pair": pair,
                "time_frame": tf,
                "signal": signal,
                "price": data["close"],
                "high": data["high"],
                "low": data["low"],
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "confidence": confidence
            })
    
    return jsonify(signals)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Dynamic port for deployment
    app.run(debug=True, host='0.0.0.0', port=port)
