"""
Utility script to load a previously trained football match model and make
predictions about whether the home team will win.  This script expects
that ``model.pkl`` (produced by ``train_model.py``) is present in the
current working directory and that it contains a dictionary with two
entries: the scikit‑learn model under the ``model`` key and the
``LabelEncoder`` used to encode team names under ``team_encoder``.

Example usage::

    python predict.py --home_team Brazil --away_team Germany \
        --neutral 0 --friendly 0

The script will output either ``Home win`` or ``Home not win``.
"""

import argparse
import joblib
import pandas as pd


from pathlib import Path

def load_model(path: str = "model.pkl"):
    """Load the serialized model and label encoder from disk.

    The path is resolved relative to this script to allow invocation from
    any working directory.
    """
    model_path = Path(__file__).resolve().parent / path
    return joblib.load(model_path)


def predict_outcome(
    model_data, home_team: str, away_team: str, neutral: int, friendly: int
) -> str:
    """Predict whether the home team wins given team names and flags.

    Parameters
    ----------
    model_data : dict
        Dictionary containing a fitted scikit‑learn model under ``model`` and
        a fitted ``LabelEncoder`` under ``team_encoder``.
    home_team : str
        Name of the home team.
    away_team : str
        Name of the away team.
    neutral : int
        1 if the match is on neutral territory, otherwise 0.
    friendly : int
        1 if the match is a friendly, otherwise 0.

    Returns
    -------
    str
        ``Home win`` if predicted to win, otherwise ``Home not win``.
    """
    model = model_data["model"]
    encoder = model_data["team_encoder"]
    # Encode teams; unknown teams will raise an error.
    home_enc = encoder.transform([home_team])[0]
    away_enc = encoder.transform([away_team])[0]
    df = pd.DataFrame(
        [[home_enc, away_enc, int(neutral), int(friendly)]],
        columns=["home_team_enc", "away_team_enc", "neutral", "is_friendly"],
    )
    pred = model.predict(df)[0]
    return "Home win" if pred == 1 else "Home not win"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict home team outcome")
    parser.add_argument("--home_team", required=True, help="Name of the home team")
    parser.add_argument("--away_team", required=True, help="Name of the away team")
    parser.add_argument(
        "--neutral",
        type=int,
        default=0,
        help="1 if match is neutral, 0 otherwise (default: 0)",
    )
    parser.add_argument(
        "--friendly",
        type=int,
        default=0,
        help="1 if match is a friendly, 0 otherwise (default: 0)",
    )
    args = parser.parse_args()
    data = load_model()
    result = predict_outcome(
        data, args.home_team, args.away_team, args.neutral, args.friendly
    )
    print(result)


if __name__ == "__main__":
    main()