from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import time
import os
import gdown
from twelvedata import TDClient
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Google Drive File ID
FILE_ID = "1x-ZeuGxIFn_3ozqe98wp7bLoKC7h_S6S"
MODEL_PATH = "forex_trading_model.pkl"

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
timeframes = ["1min", "5min", "15min", "30min", "1h"]

# Function to fetch live forex data
def fetch_live_forex_data():
    forex_data = {}
    for pair in currency_pairs:
        forex_data[pair] = {}
        for tf in timeframes:
            try:
                data = td.time_series(symbol=pair, interval=tf, outputsize=60, timezone="UTC").as_pandas()
                if not data.empty:
                    forex_data[pair][tf] = data
            except Exception as e:
                logging.error(f"Error fetching {pair} at {tf}: {e}")
                forex_data[pair][tf] = None
            time.sleep(1)
    return forex_data

# Function to prepare data and generate signals
def generate_signal(model, data, symbol, interval):
    if data is None or data.empty:
        return None
    
    # Ensure the model receives correct feature set
    expected_features = model.feature_names_in_
    latest_data = data.iloc[-1]
    input_features = np.array([latest_data.get(f, 0) for f in expected_features]).reshape(1, -1)
    
    # Generate prediction
    prediction = model.predict(input_features)[0]
    signal = "Buy" if prediction == 1 else "Sell"
    entry_price = round(latest_data["close"], 5)
    stop_loss = round(entry_price * 0.995, 5) if signal == "Buy" else round(entry_price * 1.005, 5)
    take_profit = round(entry_price * 1.005, 5) if signal == "Buy" else round(entry_price * 0.995, 5)
    
    return symbol, interval, signal, entry_price, stop_loss, take_profit

@app.route('/')
def home():
    return jsonify({"message": "Forex Trading Signal API is Running!"})

@app.route('/api/predict', methods=['GET'])
def get_trading_signals():
    live_data = fetch_live_forex_data()
    signals = []
    
    for symbol, timeframes_data in live_data.items():
        for interval, data in timeframes_data.items():
            result = generate_signal(model, data, symbol, interval)
            if result:
                symbol, interval, signal, entry_price, stop_loss, take_profit = result
                signals.append({
                    "currency_pair": symbol,
                    "time_frame": interval,
                    "signal": signal,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit
                })
    
    return jsonify(signals)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Dynamic port for deployment
    app.run(debug=True, host='0.0.0.0', port=port)
