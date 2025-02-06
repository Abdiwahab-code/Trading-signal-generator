from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import yfinance as yf

# Load the saved historical model
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

# Function to fetch live forex prices for multiple time frames
def fetch_live_forex_data():
    currency_pairs = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X"]
    time_frames = {"1m": "1d", "5m": "5d", "15m": "1mo", "1h": "3mo", "4h": "6mo", "1d": "1y"}  # Time frame mappings
    
    forex_data = {}

    for pair in currency_pairs:
        forex_data[pair] = {}

        for tf, period in time_frames.items():
            try:
                data = yf.download(tickers=pair, period=period, interval=tf)  # Fetch data for each time frame
                if not data.empty:
                    close_price = round(data["Close"].iloc[-1], 5)  # Latest close price
                    high_price = round(data["High"].iloc[-1], 5)    # Latest high price
                    low_price = round(data["Low"].iloc[-1], 5)      # Latest low price
                    forex_data[pair][tf] = {
                        "close": close_price,
                        "high": high_price,
                        "low": low_price
                    }
            except Exception as e:
                print(f"Error fetching {pair} at {tf}: {e}")
                forex_data[pair][tf] = None  # Handle errors gracefully

    return forex_data

@app.route('/api/get-trading-signals', methods=['GET'])
def get_trading_signals():
    # Fetch live forex data
    live_prices = fetch_live_forex_data()
    
    signals = []
    
    for pair, time_frames in live_prices.items():
        for tf, data in time_frames.items():
            if data is None:
                continue  # Skip if no data

            # Merge live data into a feature set for prediction
            features = np.array([data["close"], data["high"], data["low"]]).reshape(1, -1)  # Modify based on model training

            if model:
                prediction = model.predict(features)[0]
                confidence = round(max(model.predict_proba(features)[0]), 2) if hasattr(model, "predict_proba") else None
            else:
                prediction, confidence = "Unknown", None
            
            # Interpret prediction
            signal = "Buy" if prediction == 1 else "Sell"
            
            # Calculate Entry, Stop Loss & Take Profit
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
    import os
    port = int(os.environ.get("PORT", 5000))  # Render provides a PORT dynamically
    app.run(debug=True, host='0.0.0.0', port=port)
