# feature_engineering.py
import numpy as np
import pandas as pd
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder


# -------------------------------------------------------------
#  Utils
# -------------------------------------------------------------
def _split_field(x):
    """
    Convert 'A-B-C' → ['A','B','C'].
    Handles NaN, None, or empty values.
    """
    if isinstance(x, str) and len(x) > 0:
        return x.split("-")
    return []


def save_encoders(encoders, path: str):
    """
    Save all encoders to a pickle file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(encoders, f)


def load_encoders(path: str):
    """
    Load previously saved encoders.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


# -------------------------------------------------------------
#  1. Construction of encoders (training)
# -------------------------------------------------------------
def build_encoders(df: pd.DataFrame):
    """
    Build all encoders: genres, cast, companies, languages, TF-IDF.
    Return a complete dictionary.
    """

    encoders = {}

    # MultiLabelBinarizer — Genres
    genres_list = df["genres"].apply(_split_field)
    mlb_genres = MultiLabelBinarizer()
    mlb_genres.fit(genres_list)
    encoders["genres"] = mlb_genres

    # Cast
    cast_list = df["credits"].apply(_split_field)
    mlb_cast = MultiLabelBinarizer()
    mlb_cast.fit(cast_list)
    encoders["credits"] = mlb_cast

    # Companies
    comp_list = df["production_companies"].apply(_split_field)
    mlb_companies = MultiLabelBinarizer()
    mlb_companies.fit(comp_list)
    encoders["companies"] = mlb_companies

    # Languages
    ohe_lang = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe_lang.fit(df[["original_language"]])
    encoders["language"] = ohe_lang

    # TF-IDF overview
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf.fit(df["overview"].fillna(""))
    encoders["tfidf"] = tfidf

    return encoders


# -------------------------------------------------------------
#  2. Transformation dataset complet (training)
# -------------------------------------------------------------
def transform_with_encoders(df: pd.DataFrame, encoders: dict):
    """
    Transform df into X,y using already trained encoders.
    """

    genres_list = df["genres"].apply(_split_field)
    cast_list = df["credits"].apply(_split_field)
    comp_list = df["production_companies"].apply(_split_field)

    # MultiLabel encoding
    genres_encoded = encoders["genres"].transform(genres_list)
    cast_encoded = encoders["credits"].transform(cast_list)
    companies_encoded = encoders["companies"].transform(comp_list)

    # Language
    lang_encoded = encoders["language"].transform(df[["original_language"]])

    # TF-IDF
    overview_encoded = encoders["tfidf"].transform(df["overview"].fillna("")).toarray()

    # Numeric
    numeric = df[["budget", "revenue", "runtime", "popularity", "vote_count"]].values

    # Concatenation
    X = np.hstack([
        numeric,
        genres_encoded,
        companies_encoded,
        cast_encoded,
        lang_encoded,
        overview_encoded
    ])

    y = df["vote_average"].values

    return X, y


# -------------------------------------------------------------
#  3. Complete training pipeline: build + transform + save
# -------------------------------------------------------------
def preprocess_data(df: pd.DataFrame, encoders_path: str | None = None):
    """
    Complete pipeline:
        - build encoders
        - transform df into X,y
        - save encoders if path provided
    """
    encoders = build_encoders(df)
    X, y = transform_with_encoders(df, encoders)

    if encoders_path is not None:
        save_encoders(encoders, encoders_path)

    return X, y, encoders


# -------------------------------------------------------------
#  4. Preparation of a single movie (prediction)
# -------------------------------------------------------------
def preprocess_single_input(movie: dict, encoders: dict):
    """
    Transform a *non-existing* movie (dict) into a vector X ready for prediction.
    """

    numeric = np.array([
        movie.get("budget", 0),
        movie.get("revenue", 0),
        movie.get("runtime", 0),
        movie.get("popularity", 0),
        movie.get("vote_count", 0),
    ]).reshape(1, -1)

    # Multilabel encodings
    genres = _split_field(movie.get("genres", ""))
    cast = _split_field(movie.get("credits", ""))
    companies = _split_field(movie.get("production_companies", ""))

    genres_enc = encoders["genres"].transform([genres])
    cast_enc = encoders["credits"].transform([cast])
    comp_enc = encoders["companies"].transform([companies])

    # Language
    lang_enc = encoders["language"].transform([[movie.get("original_language", "en")]])

    # TF-IDF overview
    overview = movie.get("overview", "")
    overview_enc = encoders["tfidf"].transform([overview]).toarray()

    # Final concatenation
    X = np.hstack([
        numeric,
        genres_enc,
        comp_enc,
        cast_enc,
        lang_enc,
        overview_enc
    ])

    return X
