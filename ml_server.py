import os
from flask import Flask, request, jsonify
import threading
import time
import json

# Optional: requests for self-ping
try:
    import requests
except ImportError:
    os.system("pip install requests")
    import requests

app = Flask(__name__)

# Config from environment
ML_SERVER_URL = os.getenv("ML_SERVER_URL")  # Your Render URL
PING_INTERVAL = 600  # 10 minutes

# Example in-memory "model" data store (replace with real ML logic if needed)
MODEL_DATA = []

def predict(seed, position):
    """Dummy prediction using seed and position"""
    import random
    random.seed(seed + position)
    return random.randint(0, 100)

def train_model(submission):
    """Store submission in MODEL_DATA, replace with real ML training"""
    MODEL_DATA.append(submission)
    print(f"New submission added: {submission}")

@app.route("/")
def home():
    return jsonify({"status": "ML server running"}), 200

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.json
    seed = data.get("seed")
    position = data.get("position")

    if seed is None or position is None:
        return jsonify({"error": "Missing seed or position"}), 400

    result = predict(seed, position)
    return jsonify({"prediction": result})

@app.route("/submit", methods=["POST"])
def submit_route():
    """Endpoint to submit results for training"""
    data = request.json
    seed = data.get("seed")
    position = data.get("position")
    outcome = data.get("outcome")  # e.g., the actual dice result

    if seed is None or position is None or outcome is None:
        return jsonify({"error": "Missing seed, position, or outcome"}), 400

    submission = {"seed": seed, "position": position, "outcome": outcome}
    train_model(submission)
    return jsonify({"status": "success", "submission": submission}), 200

def self_ping():
    """Keep server awake on free-tier Render"""
    if not ML_SERVER_URL:
        print("ML_SERVER_URL not set, self-ping disabled")
        return
    while True:
        try:
            r = requests.get(ML_SERVER_URL)
            print(f"Self-ping status: {r.status_code}")
        except Exception as e:
            print(f"Self-ping failed: {e}")
        time.sleep(PING_INTERVAL)

if __name__ == "__main__":
    # Start self-ping thread
    if ML_SERVER_URL:
        threading.Thread(target=self_ping, daemon=True).start()

    # Use port from Render environment
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
