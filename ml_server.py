import os
import threading
import time
import json
from flask import Flask, request, jsonify

# Optional: requests for self-ping
try:
    import requests
except ImportError:
    os.system("pip install requests")
    import requests

app = Flask(__name__)

# Config from environment
ML_SERVER_URL = os.getenv("ML_SERVER_URL")  # Your Render URL
PING_INTERVAL = 10 * 60  # 10 minutes

# In-memory storage of submissions
MODEL_DATA = []

# --------------------
# ML Utilities
# --------------------
def weighted_prediction(submissions):
    """
    Compute a simple weighted prediction of bomb positions based on submissions.
    Each submission has a weight (0-1) and contributes to the final prediction.
    """
    if not submissions:
        return "No data yet"

    # For simplicity, assume we predict next bomb position 0-100
    weighted_counts = [0] * 101  # positions 0-100
    total_weight = 0

    for sub in submissions:
        outcome = sub.get("outcome")
        weight = sub.get("weight", 1)
        if isinstance(outcome, int) and 0 <= outcome <= 100:
            weighted_counts[outcome] += weight
            total_weight += weight

    if total_weight == 0:
        return "Insufficient weighted data"

    # Return position with highest weighted count
    prediction = weighted_counts.index(max(weighted_counts))
    return prediction

def train_model(submission):
    """Add submission to MODEL_DATA with weight"""
    MODEL_DATA.append(submission)
    print(f"New submission added: {submission}")

# --------------------
# Flask Routes
# --------------------
@app.route("/")
def home():
    return jsonify({"status": "ML server running", "submissions": len(MODEL_DATA)}), 200

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.json
    submissions = data.get("submissions")  # list of submissions with weights
    if not submissions:
        return jsonify({"error": "No submissions provided"}), 400

    prediction = weighted_prediction(submissions)
    return jsonify({"prediction": prediction})

@app.route("/submit", methods=["POST"])
def submit_route():
    """
    Accepts a single submission for training:
    {seed, position, outcome, weight(optional)}
    """
    data = request.json
    seed = data.get("seed")
    position = data.get("position")
    outcome = data.get("outcome")
    weight = data.get("weight", 1)

    if seed is None or position is None or outcome is None:
        return jsonify({"error": "Missing seed, position, or outcome"}), 400

    submission = {"seed": seed, "position": position, "outcome": outcome, "weight": weight}
    train_model(submission)
    return jsonify({"status": "success", "submission": submission}), 200

# --------------------
# Self-Ping to Stay Awake
# --------------------
def self_ping():
    """
    Periodically ping the ML server to keep it awake.
    Uses exponential backoff if ping fails.
    """
    if not ML_SERVER_URL:
        print("ML_SERVER_URL not set, self-ping disabled")
        return

    interval = PING_INTERVAL
    while True:
        try:
            r = requests.get(ML_SERVER_URL)
            print(f"Self-ping status: {r.status_code}")
            interval = PING_INTERVAL  # reset interval on success
        except Exception as e:
            print(f"Self-ping failed: {e}")
            # Backoff: double interval up to 1 hour
            interval = min(interval * 2, 3600)
        time.sleep(interval)

# --------------------
# Main
# --------------------
if __name__ == "__main__":
    # Start self-ping thread
    if ML_SERVER_URL:
        threading.Thread(target=self_ping, daemon=True).start()

    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
