"""
Simple Flask API to serve predictions for international football matches.

This script loads a previously trained model from ``model.pkl`` (see
``train_model.py``) and exposes an HTTP endpoint at ``/predict``.  The
endpoint accepts POST requests with a JSON body containing the
``home_team``, ``away_team``, ``neutral`` and ``friendly`` fields and
returns a JSON response with a ``prediction`` key.  The prediction
value is ``home_win`` or ``home_not_win``.

Note: The Flask dependency is listed in ``requirements.txt`` but may not
be installed in this environment.  To run the API locally, install
dependencies and then execute ``python app.py``.
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and label encoder once at startup.
from pathlib import Path
# Resolve model path relative to this file so the app can be started
# from any working directory.
MODEL_DATA = joblib.load(Path(__file__).resolve().parent / "model.pkl")
MODEL = MODEL_DATA["model"]
ENCODER = MODEL_DATA["team_encoder"]


@app.route("/predict", methods=["POST"])
def predict() -> tuple:
    """Predict outcome of a football match for the home team.

    Expected JSON payload:
    {
      "home_team": "Brazil",
      "away_team": "Germany",
      "neutral": 0,
      "friendly": 0
    }
    """
    content = request.json or {}
    try:
        home_team = content["home_team"]
        away_team = content["away_team"]
    except KeyError:
        return jsonify({"error": "home_team and away_team are required"}), 400
    neutral = int(content.get("neutral", 0))
    friendly = int(content.get("friendly", 0))
    # Encode teams; error if unknown team.
    try:
        home_enc = ENCODER.transform([home_team])[0]
        away_enc = ENCODER.transform([away_team])[0]
    except Exception:
        return jsonify({"error": "Unknown team name"}), 400
    X = pd.DataFrame(
        [[home_enc, away_enc, neutral, friendly]],
        columns=["home_team_enc", "away_team_enc", "neutral", "is_friendly"],
    )
    pred = MODEL.predict(X)[0]
    result = "home_win" if pred == 1 else "home_not_win"
    return jsonify({"prediction": result})


if __name__ == "__main__":
    # Start the Flask development server.
    app.run(host="0.0.0.0", port=5000, debug=True)