# Soccer Match Outcome Prediction

This project demonstrates how to build and deploy a simple machine‑learning
model to predict whether the home team will win an international
football match.  It uses a freely available dataset of historic
international match results from 1872 onwards, which can be
downloaded directly from a public GitHub repository【271827671255260†L37-L55】.

## Dataset

The raw data is stored in ``data/results.csv``.  Each row in the
dataset contains information about a single match, including the
date, home team, away team, final scores, tournament name, city,
country and whether the match was played on neutral ground【271827671255260†L37-L55】.  The
CSV file was downloaded from `martj42/international_results` on GitHub
using the `raw.githubusercontent.com` URL and saved locally.

## Project Structure

```
soccer-ml-project/
├── data/
│   └── results.csv       # raw match results dataset
├── train_model.py        # script to train a random forest classifier
├── predict.py            # command‑line script to load the model and make predictions
├── app.py                # Flask API exposing a /predict endpoint
├── model.pkl             # saved model and label encoder (generated after training)
├── requirements.txt      # Python dependencies
└── README.md             # project documentation
```

## Training the Model

Run the training script to build a classifier and save it to
``model.pkl``:

```bash
python train_model.py
```

The script performs the following steps:

1. Loads the data from ``data/results.csv`` and removes any matches without
   recorded scores.
2. Creates a binary target variable indicating whether the home team won.
3. Encodes team names into numerical identifiers using a single
   ``LabelEncoder`` so the same mapping applies to both home and away teams.
4. Encodes whether the match is a friendly and whether it is played on
   neutral ground.
5. Splits the data into training and testing sets and trains a
   random forest classifier.
6. Evaluates the model on the test set and prints the validation accuracy.
7. Saves the trained model and encoder to ``model.pkl`` for later use.

## Making Predictions from the Command Line

After training, you can make predictions for a new matchup using
``predict.py``.  For example, to predict whether Brazil would beat
Germany in a non‑friendly home match:

```bash
python predict.py --home_team Brazil --away_team Germany --neutral 0 --friendly 0
```

The script will output either ``Home win`` or ``Home not win``.

## Running the API

If you wish to deploy the model as a web service, install the
dependencies in ``requirements.txt`` and run ``app.py``.  The Flask
application exposes a ``POST /predict`` endpoint that accepts a JSON
payload with the following fields:

```json
{
  "home_team": "Brazil",
  "away_team": "Germany",
  "neutral": 0,
  "friendly": 0
}
```

It returns a JSON response containing the prediction (``home_win`` or
``home_not_win``).  Example:

```bash
curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d '{"home_team": "Brazil", "away_team": "Germany", "neutral": 0, "friendly": 0}'

{
  "prediction": "home_win"
}
```

## License

This project uses the international football results dataset released
under the Creative Commons CC0 1.0 Universal Public Domain Dedication【271827671255260†L37-L55】.
All code in this repository is provided under the MIT License.
