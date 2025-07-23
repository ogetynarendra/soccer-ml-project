"""
Train a simple machine‑learning model to predict home‑team victory in
international football matches. The script loads the raw results dataset
from the ``data/results.csv`` file (downloaded from a public GitHub
repository), performs basic feature engineering and preprocessing, and
trains a random forest classifier.  It also persists the fitted model
and the label encoder to ``model.pkl`` so they can be reused for
predictions or served via an API.

The underlying dataset, described in a blog post about football data
analysis, contains match results from as early as 1872 and includes
columns such as date, home team, away team, scores, tournament and
whether the match was played on neutral ground【271827671255260†L37-L55】.

Example usage::

    python train_model.py

This will print the validation accuracy and write ``model.pkl`` to
the current working directory.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


def main() -> None:
    """Load data, train a classifier and persist the model."""
    # Load the soccer results dataset.  The CSV includes columns such as
    # date, home_team, away_team, home_score, away_score, tournament,
    # city, country and neutral【271827671255260†L53-L55】.  Some historic
    # matches may not have scores recorded; drop these rows.
    # Resolve the path to the dataset relative to this script.  Using
    # ``__file__`` ensures the code works regardless of the working
    # directory.
    from pathlib import Path
    data_path = Path(__file__).resolve().parent / "data" / "results.csv"
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["home_score", "away_score"]).copy()

    # Create a binary target: 1 if the home team wins, 0 otherwise.
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    # Encode team names into numerical identifiers.  One label encoder
    # handles both home and away teams so identical team names map to
    # the same integer across both columns.
    team_encoder = LabelEncoder()
    all_teams = pd.concat([df["home_team"], df["away_team"]], axis=0)
    team_encoder.fit(all_teams)
    df["home_team_enc"] = team_encoder.transform(df["home_team"])
    df["away_team_enc"] = team_encoder.transform(df["away_team"])

    # Encode whether the match was played on neutral territory (1 for
    # neutral, 0 for home advantage) and whether it was a friendly.
    df["neutral"] = df["neutral"].astype(int)
    df["is_friendly"] = (df["tournament"] == "Friendly").astype(int)

    # Define features and target.
    features = ["home_team_enc", "away_team_enc", "neutral", "is_friendly"]
    X = df[features]
    y = df["home_win"]

    # Split into train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a random forest classifier.  Random forests handle
    # categorical codes and non‑linear relationships reasonably well.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on the hold‑out set and print accuracy.
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation accuracy: {acc:.3f}")

    # Persist the model and encoder for downstream use (e.g. API).
    joblib.dump({"model": model, "team_encoder": team_encoder}, "model.pkl")
    print("Saved trained model to model.pkl")


if __name__ == "__main__":
    main()