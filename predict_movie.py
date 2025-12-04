# predict_movie.py
import xgboost as xgb
import numpy as np

from feature_engineering import load_encoders, preprocess_single_input


def predict_movie(movie: dict,
                  model_path="models/movie_xgb.json",
                  encoders_path="models/encoders.pkl"):

    # Load model and encoders
    booster = xgb.Booster()
    booster.load_model(model_path)

    encoders = load_encoders(encoders_path)

    # Transform the movie into a feature vector
    X = preprocess_single_input(movie, encoders)
    dmatrix = xgb.DMatrix(X)

    # Prediction
    y_pred = booster.predict(dmatrix)
    return float(y_pred[0])


if __name__ == "__main__":
    # Example of a fictitious movie
    movie = {
        "title": "My Future Ghost",
        "genres": "Action-Science Fiction",
        "original_language": "en",
        "overview": "In a distant future, a lone hero must save humanity from a mysterious AI.",
        "popularity": 500.0,
        "production_companies": "Marvel Studios-Disney",
        "budget": 200_000_000,
        "revenue": 800_000,
        "runtime": 135,
        "vote_average": 0.0,          # unknown, ignored in features
        "vote_count": 500,
        "credits": "Vin Diesel-Famous Actor 2-Famous Actor 3",
    }

    predicted_rating = predict_movie(movie)
    print(f"Predicted vote_average for '{movie['title']}': {predicted_rating:.2f}")
