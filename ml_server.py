from flask import Flask, request, jsonify
import random

app = Flask(__name__)

def predict_next(last_positions):
    # Simple prediction, replace later with real ML
    board_size = 25
    predictions = [{"position": i, "probability": random.random()} for i in range(1, board_size+1)]
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
    print(f"Training ML: {data}")
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
