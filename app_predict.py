# app_predict.py
import xgboost as xgb
import pickle
import pandas as pd

from feature_engineering import preprocess_single_input, load_encoders

MODEL_PATH = "models/movie_xgb.json"
ENCODERS_PATH = "models/encoders.pkl"


def ask_user_movie():
    print("\n=== Fill in the film's characteristics ===\n")

    movie = {
        "title": input("Title of the movie: ").strip(),
        "genres": input("Genres (e.g., Action-Comedy): ").strip(),
        "original_language": input("Original language (e.g., en, fr, es): ").strip(),
        "overview": input("Overview of the movie: ").strip(),
        "popularity": float(input("Popularity (number): ")),
        "production_companies": input("Studios (e.g., Marvel-Disney): ").strip(),
        "budget": float(input("Budget ($): ")),
        "revenue": float(input("Revenue ($): ")),
        "runtime": float(input("Duration (minutes): ")),
        "vote_count": float(input("Estimated vote count: ")),
        "credits": input("Main actors (e.g., Brad Pitt-Leo DiCaprio): ").strip()
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

    print("\n===== PREDICTION =====")
    print(f"Title: {movie['title']}")
    print(f"Predicted rating: ⭐ {pred:.2f} / 10\n")