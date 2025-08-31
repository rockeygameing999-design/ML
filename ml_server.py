import os
import threading
import time
import requests
from flask import Flask, request, jsonify

# Initialize Flask
app = Flask(__name__)

# Environment variables
ML_SERVER_URL = os.getenv("ML_SERVER_URL")  # Your Render public URL

if not ML_SERVER_URL:
    print("❌ Warning: ML_SERVER_URL not set in .env, self-ping disabled")

# Dummy prediction function (replace with your ML logic)
def predict(data):
    # Example: just returns data for now
    return {"prediction": "example", "input": data}

# Flask routes
@app.route("/")
def home():
    return "ML Server is alive! 🚀"

@app.route("/predict", methods=["POST"])
def handle_predict():
    try:
        data = request.json
        result = predict(data)
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Self-ping to stay awake on Render free tier
def self_ping():
    while True:
        if ML_SERVER_URL:
            try:
                print("🔔 Self-ping to keep server awake...")
                requests.get(ML_SERVER_URL)
            except Exception as e:
                print("❌ Self-ping failed:", e)
        time.sleep(10 * 60)  # Ping every 10 minutes

if __name__ == "__main__":
    # Start self-ping thread
    ping_thread = threading.Thread(target=self_ping, daemon=True)
    ping_thread.start()

    # Run Flask app
    app.run(host="0.0.0.0", port=10000)
