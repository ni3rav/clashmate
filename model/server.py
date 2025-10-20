from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Load model and encoders
try:
    with open("elixir_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("type_encoder.pkl", "rb") as f:
        type_encoder = pickle.load(f)
    with open("feature_medians.pkl", "rb") as f:
        feature_medians = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model files not found. Please run train_model.py first!")
    model = None

# Load card data for comparison
df = pd.read_csv("cards.csv")


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/styles.css")
def styles():
    return send_from_directory(".", "styles.css")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return (
            jsonify({"error": "Model not trained yet. Run train_model.py first"}),
            500,
        )

    data = request.json

    # Extract features from request
    card_type = data.get("type", "Troop")
    hitpoints = float(data.get("hitpoints", 0)) or feature_medians.get(
        "hitpoints_clean", 0
    )
    damage = float(data.get("damage", 0)) or feature_medians.get("damage_clean", 0)
    hit_speed = float(data.get("hitSpeed", 0)) or feature_medians.get(
        "hitSpeed_clean", 0
    )
    dps = float(data.get("dps", 0)) or feature_medians.get("dps_clean", 0)
    range_val = float(data.get("range", 0)) or feature_medians.get("range_clean", 0)
    count = float(data.get("count", 0)) or feature_medians.get("count_clean", 0)

    # Encode type
    try:
        type_encoded = type_encoder.transform([card_type])[0]
    except:
        type_encoded = 0

    has_area_damage = 0
    has_spawned_unit = 0
    features = np.array(
        [
            [
                type_encoded,
                hitpoints,
                damage,
                hit_speed,
                dps,
                range_val,
                count,
                has_area_damage,
                has_spawned_unit,
            ]
        ]
    )

    # Predict
    prediction = model.predict(features)[0]

    # Get prediction interval (confidence range)
    predictions = [tree.predict(features)[0] for tree in model.estimators_]
    std = np.std(predictions)
    confidence_lower = max(1, prediction - std)
    confidence_upper = min(10, prediction + std)

    return jsonify(
        {
            "predicted_elixir": round(prediction, 2),
            "confidence_lower": round(confidence_lower, 2),
            "confidence_upper": round(confidence_upper, 2),
            "message": f"Estimated elixir cost: {prediction:.1f}",
        }
    )


@app.route("/cards", methods=["GET"])
def get_cards():
    """Return all cards data for visualization"""
    # Clean and return card data
    cards_data = []
    for _, row in df.iterrows():
        elixir = row.get("elixir", "")
        if elixir and elixir != "N/A":
            import re

            match = re.search(r"(\d+)", str(elixir))
            if match:
                cards_data.append(
                    {
                        "name": row.get("name", ""),
                        "type": row.get("type", ""),
                        "elixir": int(match.group(1)),
                    }
                )

    return jsonify(cards_data)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    print("\nStarting ClashMate Server...")
    print("Server running at http://localhost:5000")
    print("\nEndpoints:")
    print("   GET  /        - Web interface")
    print("   POST /predict - Predict elixir cost")
    print("   GET  /cards   - Get all cards data")
    print("   GET  /health  - Health check")
    print()
    app.run(debug=True, host="0.0.0.0", port=5000)
