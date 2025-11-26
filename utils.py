import pandas as pd
import numpy as np

# Utility functions

def clean_numeric(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def drop_invalid(df, conditions):
    for condition in conditions:
        df = df.loc[condition]  
    return df


def require_text_fields(df, fields):
    return df.dropna(subset=fields)