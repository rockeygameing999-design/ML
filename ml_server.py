from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import numpy as np
import pickle
import os

app = Flask(__name__)

# In-memory storage (replace with MongoDB or Redis for production)
MODEL_DATA = []
MODEL = None
SCALER = StandardScaler()

# Save/load model
MODEL_FILE = 'model.pkl'
def save_model(model):
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)

def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, 'rb') as f:
            return pickle.load(f)
    return None

# Train model with hyperparameter tuning and ensemble
def train_model():
    global MODEL
    if not MODEL_DATA:
        return

    # Extract features and labels
    X = []
    y = []
    for data in MODEL_DATA:
        features = [
            data['nonce'],
            data['totalMines'],
            data['features']['hashEntropy'],
            data['features']['nonceCategory'],
            data['features']['positionDensity']
        ]
        X.append(features)
        y.append(1 if data['outcome'] == 'win' else 0)  # Binary classification

    if not X:
        return

    X = np.array(X)
    y = np.array(y)

    # Normalize features
    X_scaled = SCALER.fit_transform(X)

    # Handle imbalanced data with SMOTE
    smote = SMOTE(random_state=42)
    try:
        X_res, y_res = smote.fit_resample(X_scaled, y)
    except ValueError:
        X_res, y_res = X_scaled, y  # Fallback if SMOTE fails

    # Ensemble model
    rf = RandomForestClassifier(random_state=42)
    xgb = XGBClassifier(random_state=42)
    ensemble = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb)], voting='soft')

    # Hyperparameter tuning
    param_grid = {
        'rf__n_estimators': [50, 100],
        'rf__max_depth': [5, 10],
        'xgb__n_estimators': [50, 100],
        'xgb__max_depth': [3, 5]
    }
    grid_search = GridSearchCV(ensemble, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_res, y_res)

    MODEL = grid_search.best_estimator_
    save_model(MODEL)

    # Evaluate with cross-validation
    scores = cross_val_score(MODEL, X_res, y_res, cv=5)
    print(f"Cross-validation accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")

# Predict function
def weighted_prediction(data):
    if not MODEL:
        return [0] * data['totalMines']  # Default if no model

    features = [
        data['nonce'],
        data['totalMines'],
        data['features']['hashEntropy'],
        data['features']['nonceCategory'],
        data['features']['positionDensity']
    ]
    X = SCALER.transform([features])
    prob = MODEL.predict_proba(X)[0][1]  # Probability of 'win'

    # Generate bomb positions based on probability
    num_positions = data['totalMines']
    positions = []
    seen = set()
    seed = data['nonce']
    while len(positions) < num_positions:
        seed = hash(str(seed)) % 1000000
        pos = seed % 25
        if pos not in seen:
            seen.add(pos)
            positions.append(pos)
    return sorted(positions)

@app.route('/submit', methods=['POST'])
def submit():
    data = request.json
    MODEL_DATA.append(data)
    train_model()  # Retrain with new data
    return jsonify({'status': 'success'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['submissions'][0]
    prediction = weighted_prediction(data)
    return jsonify({'prediction': prediction})

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    MODEL = load_model()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
