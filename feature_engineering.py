import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
import numpy as np

def preprocess_data(df):
    # ----- CLEAN SPLIT -----
    def split_field(x):
        if isinstance(x, str) and len(x) > 0:
            return x.split("-")
        return []

    df["genres"] = df["genres"].apply(split_field)
    df["production_companies"] = df["production_companies"].apply(split_field)
    df["credits"] = df["credits"].apply(split_field)

    # ----- ENCODING -----
    mlb_genres = MultiLabelBinarizer()
    genres_encoded = mlb_genres.fit_transform(df["genres"])

    mlb_comp = MultiLabelBinarizer()
    companies_encoded = mlb_comp.fit_transform(df["production_companies"])

    mlb_cast = MultiLabelBinarizer()
    cast_encoded = mlb_cast.fit_transform(df["credits"])

    # ----- LANGUAGE -----
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    lang_encoded = ohe.fit_transform(df[["original_language"]])

    # ----- OVERVIEW TF-IDF -----
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    overview_vec = tfidf.fit_transform(df["overview"].fillna("")).toarray()

    # ----- NUMERIC FEATURES -----
    num = df[["budget", "revenue", "runtime", "popularity", "vote_count"]].values

    # ----- CONCATENATION -----
    X = np.hstack([
        num,
        genres_encoded,
        companies_encoded,
        cast_encoded,
        lang_encoded,
        overview_vec
    ])

    y = df["vote_average"].values

    return X, y
