from flask import Flask, request, jsonify
import random
import os

app = Flask(__name__)

# Dummy prediction function (replace with real ML later)
def predict_next(last_positions):
    board_size = 25  # adjust for your Mine game
    predictions = [{"position": i, "probability": random.random()} for i in range(1, board_size + 1)]
    total = sum(p["probability"] for p in predictions)
    for p in predictions:
        p["probability"] /= total
    return predictions

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    last_positions = data.get("last_positions", [])
    return jsonify({"predictions": predict_next(last_positions)})

@app.route("/train", methods=["POST"])
def train():
    data = request.get_json()
    user_id = data.get("user_id")
    boom_position = data.get("boom_position")
    # Here you can train your ML model
    print(f"Training ML with user {user_id}, boom_position {boom_position}")
    return jsonify({"status": "success"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render’s port
    app.run(host="0.0.0.0", port=port)
