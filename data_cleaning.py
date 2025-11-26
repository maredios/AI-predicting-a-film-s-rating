import pandas as pd
from config import ALLOWED_COLUMNS, MIN_BUDGET, MIN_REVENUE, MIN_VOTE_COUNT, RAW_DATASET_PATH, CLEAN_DATASET_PATH
from utils import clean_numeric, drop_invalid, require_text_fields

# 1. Load raw dataset and clean it

def clean_dataset(input_path=RAW_DATASET_PATH, output_path=CLEAN_DATASET_PATH):

    df = pd.read_csv(input_path)

    df = df[ALLOWED_COLUMNS].copy()

    numeric_columns = ["popularity", "budget", "revenue", "runtime", "vote_count"]
    df = clean_numeric(df, numeric_columns)

    df = drop_invalid(df, [
        (df["budget"].notna()) & (df["budget"] >= MIN_BUDGET),
        (df["revenue"].notna()) & (df["revenue"] >= MIN_REVENUE),
        df["vote_average"].notna(),
        (df["vote_count"].notna()) & (df["vote_count"] >= MIN_VOTE_COUNT)
    ])

    text_fields = [
        "title",
        "genres",
        "original_language",
        "overview",
        "production_companies",
        "credits"
    ]

    df = require_text_fields(df, text_fields)

    df = df.reset_index(drop=True)

    df.to_csv(output_path, index=False)
    print(f"Clean dataset saved to {output_path}")

if __name__ == "__main__":
    clean_dataset()