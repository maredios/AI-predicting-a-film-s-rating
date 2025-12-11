# predict_cli.py
import xgboost as xgb
import numpy as np

from feature_engineering import load_encoders, preprocess_single_input


def predict_movie(movie: dict,
                  model_path="models/movie_xgb.json",
                  encoders_path="models/encoders.pkl"):

    booster = xgb.Booster()
    booster.load_model(model_path)

    encoders = load_encoders(encoders_path)

    X = preprocess_single_input(movie, encoders)
    dmatrix = xgb.DMatrix(X)

    y_pred = booster.predict(dmatrix)
    return float(y_pred[0])


if __name__ == "__main__":

    print("\n=== MOVIE RATING PREDICTOR ===\n")

    title = input("Movie title: ")

    genres = input("Genres (format: Action-Drama-SciFi): ")
    original_language = input("Original language (ex: en): ")

    overview = input("Overview (short description): ")

    production_companies = input("Production companies (format: Marvel-Disney): ")
    credits = input("Cast (format: Actor1-Actor2-Actor3): ")

    # ----- NUMERIC INPUTS -----
    budget = float(input("Budget (number): "))
    revenue = float(input("Revenue (number): "))
    runtime = float(input("Runtime in minutes: "))
    popularity = float(input("Popularity score: "))
    vote_count = float(input("Vote count (expected audience size): "))

    # Build movie dict
    movie = {
        "title": title,
        "genres": genres,
        "original_language": original_language,
        "overview": overview,
        "popularity": popularity,
        "production_companies": production_companies,
        "budget": budget,
        "revenue": revenue,
        "runtime": runtime,
        "vote_average": 0.0,   # ignored
        "vote_count": vote_count,
        "credits": credits,
    }

    # Predict
    predicted = predict_movie(movie)
    print(f"\n⭐ Predicted rating for '{title}': {predicted:.2f}\n")
