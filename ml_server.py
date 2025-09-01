from flask import Flask, request, jsonify
import numpy as np
import pickle
import os
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# In-memory storage (replace with MongoDB for production)
MODEL_DATA = []
MODEL = None
SCALER = None
USE_ML = True  # Flag to use ML if libraries load

# Try importing ML dependencies with fallback
try:
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    from xgboost import XGBClassifier
    from imblearn.over_sampling import SMOTE
    SCALER = StandardScaler()
    logger.info("ML libraries loaded successfully")
except ImportError as e:
    logger.warning(f"ML libraries not available: {e}. Falling back to simple prediction. Install with: pip install scikit-learn xgboost imbalanced-learn numpy")
    USE_ML = False

# Save/load model
MODEL_FILE = 'model.pkl'
def save_model(model):
    try:
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)
        logger.info("Model saved successfully")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")

def load_model():
    try:
        if os.path.exists(MODEL_FILE):
            with open(MODEL_FILE, 'rb') as f:
                model = pickle.load(f)
                logger.info("Model loaded successfully")
                return model
        return None
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

# Train model with hyperparameter tuning and ensemble (only if USE_ML)
def train_model():
    global MODEL
    if not USE_ML or not MODEL_DATA:
        logger.warning("No ML libraries or data available for training")
        return

    # Extract features and labels
    X = []
    y = []
    for data in MODEL_DATA:
        features = [
            data['nonce'],
            data['totalMines'],
            data.get('features', {}).get('hashEntropy', 0),
            data.get('features', {}).get('nonceCategory', 0),
            data.get('features', {}).get('positionDensity', 0)
        ]
        X.append(features)
        y.append(1 if data['outcome'] == 'win' else 0)  # Binary classification

    if not X:
        logger.warning("No valid features extracted")
        return

    X = np.array(X)
    y = np.array(y)

    # Normalize features
    try:
        X_scaled = SCALER.fit_transform(X)
    except Exception as e:
        logger.error(f"Feature normalization failed: {e}")
        return

    # Handle imbalanced data with SMOTE
    try:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_scaled, y)
    except ValueError as e:
        logger.warning(f"SMOTE failed: {e}. Using original data")
        X_res, y_res = X_scaled, y

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
    try:
        grid_search = GridSearchCV(ensemble, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_res, y_res)
        MODEL = grid_search.best_estimator_
        save_model(MODEL)

        # Evaluate with cross-validation
        scores = cross_val_score(MODEL, X_res, y_res, cv=5)
        logger.info(f"Cross-validation accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
    except Exception as e:
        logger.error(f"Model training failed: {e}")

# Predict function (ML if available, else simple weighted)
def weighted_prediction(data):
    if USE_ML and MODEL:
        features = [
            data['nonce'],
            data['totalMines'],
            data.get('features', {}).get('hashEntropy', 0),
            data.get('features', {}).get('nonceCategory', 0),
            data.get('features', {}).get('positionDensity', 0)
        ]
        try:
            X = SCALER.transform([features])
            prob = MODEL.predict_proba(X)[0][1]  # Probability of 'win'
            logger.info(f"ML prediction probability: {prob}")
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            prob = 0.5  # Fallback
    else:
        prob = 0.5  # Simple fallback

    # Generate bomb positions based on probability and data
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
    logger.info(f"Generated positions: {positions}")
    return sorted(positions)

@app.route('/submit', methods=['POST'])
def submit():
    try:
        data = request.json
        MODEL_DATA.append(data)
        train_model()  # Retrain with new data
        logger.info("Submission processed successfully")
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Submit endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['submissions'][0]
        prediction = weighted_prediction(data)
        logger.info(f"Prediction generated: {prediction}")
        return jsonify({'prediction': prediction})
    except Exception as e:
        logger.error(f"Predict endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/ping', methods=['GET'])
def ping():
    logger.info("Ping received")
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    MODEL = load_model()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
