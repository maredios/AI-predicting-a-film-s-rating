# app_predict.py
import xgboost as xgb
import pickle
import pandas as pd

from feature_engineering import preprocess_single_input, load_encoders

MODEL_PATH = "models/movie_xgb.json"
ENCODERS_PATH = "models/encoders.pkl"


def ask_user_movie():
    print("\n=== Remplis les caractéristiques du film ===\n")

    movie = {
        "title": input("Titre du film : ").strip(),
        "genres": input("Genres (ex: Action-Comedy) : ").strip(),
        "original_language": input("Langue originale (ex: en, fr, es) : ").strip(),
        "overview": input("Résumé du film : ").strip(),
        "popularity": float(input("Popularité (nombre) : ")),
        "production_companies": input("Studios (ex: Marvel-Disney) : ").strip(),
        "budget": float(input("Budget ($) : ")),
        "revenue": float(input("Revenus ($) : ")),
        "runtime": float(input("Durée (minutes) : ")),
        "vote_count": float(input("Vote count estimé : ")),
        "credits": input("Acteurs principaux (ex: Brad Pitt-Leo DiCaprio) : ").strip()
    }

    return movie


def predict_movie(movie):
    # Load model + encoders
    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)

    encoders = load_encoders(ENCODERS_PATH)

    # Process movie
    X = preprocess_single_input(movie, encoders)
    dmat = xgb.DMatrix(X)

    pred = booster.predict(dmat)[0]
    return pred


if __name__ == "__main__":
    movie = ask_user_movie()
    pred = predict_movie(movie)

    print("\n===== PRÉDICTION =====")
    print(f"Titre : {movie['title']}")
    print(f"Note prédite : ⭐ {pred:.2f} / 10\n")
