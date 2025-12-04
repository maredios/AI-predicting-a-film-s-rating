import os
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

from feature_engineering import build_encoders, preprocess_data


def load_encoders():
    """Load the encoders saved during preprocessing."""
    with open("models/encoders.pkl", "rb") as f:
        enc = pickle.load(f)
    return enc


def get_feature_names(encoders, df):
    """
    Generate real feature names in correct column order,
    matching the structure of preprocess_data().
    """

    feature_names = []

    # 1) NUMERICAL
    feature_names += ["budget", "revenue", "runtime", "popularity", "vote_count"]

    # 2) Genres
    feature_names += [f"genre_{g}" for g in encoders["genres"].classes_]

    # 3) Companies
    feature_names += [f"company_{c}" for c in encoders["companies"].classes_]

    # 4) Cast
    feature_names += [f"cast_{c}" for c in encoders["credits"].classes_]

    # 5) Languages
    feature_names += [f"lang_{l}" for l in encoders["language"].categories_[0]]

    # 6) TF-IDF words
    tfidf_words = encoders["tfidf"].get_feature_names_out()
    feature_names += [f"tfidf_{w}" for w in tfidf_words]

    return feature_names


def compute_feature_importance():
    print("Loading clean dataset...")
    df = pd.read_csv("data/movies_dataset_clean.csv")

    print("Loading encoders...")
    encoders = load_encoders()

    print("Building feature name space...")
    feature_names = get_feature_names(encoders, df)

    print("Loading trained model...")
    model = xgb.Booster()
    model.load_model("models/movie_xgb.json")

    print("Extracting feature importances...")
    scores = model.get_score(importance_type="gain")

    # Convert from XGBoost index → sorted list
    importances = []
    for key, val in scores.items():
        idx = int(key.replace("f", ""))
        importances.append((feature_names[idx], val))

    # Sort by importance
    importances = sorted(importances, key=lambda x: x[1], reverse=True)

    # Save to CSV
    out_df = pd.DataFrame(importances, columns=["feature", "importance"])
    out_df.to_csv("feature_importance_words.csv", index=False)

    print("\nFeature importance saved to feature_importance_words.csv")
    print("\nTop 20 features:")
    print(out_df.head(20))


if __name__ == "__main__":
    compute_feature_importance()
