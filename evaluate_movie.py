# evaluate_model.py
import pandas as pd
import xgboost as xgb
import pickle

from feature_engineering import preprocess_single_input, load_encoders


MODEL_PATH = "models/movie_xgb.json"
ENCODERS_PATH = "models/encoders.pkl"
DATASET_PATH = "data/movies_dataset_clean.csv"


def evaluate_existing_movie(title):
    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    # Check if movie exists
    row = df[df["title"].str.lower() == title.lower()]
    if row.empty:
        print(f"❌ Film introuvable dans le dataset : {title}")
        return

    movie = row.iloc[0].to_dict()
    true_rating = movie["vote_average"]

    # Load model + encoders
    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)

    encoders = load_encoders(ENCODERS_PATH)

    # Prepare movie for prediction
    X = preprocess_single_input(movie, encoders)
    dmatrix = xgb.DMatrix(X)

    pred = booster.predict(dmatrix)[0]

    print("\n===== EVALUATION DU FILM =====")
    print(f"Titre : {title}")
    print(f"Note réelle      : {true_rating:.2f}")
    print(f"Note prédite     : {pred:.2f}")
    print(f"Erreur (RMSE-like): {abs(pred - true_rating):.3f}\n")


if __name__ == "__main__":
    print("=== Tester un film ===")
    film = input("Titre du film : ").strip()
    evaluate_existing_movie(film)
