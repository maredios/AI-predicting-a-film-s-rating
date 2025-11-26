# Configuration file


# Columns allowed in the dataset
ALLOWED_COLUMNS = [
"title",
"genres",
"original_language",
"overview",
"popularity",
"production_companies",
"budget",
"revenue",
"runtime",
"vote_average",
"vote_count",
"credits"
]


# Minimum allowed values
MIN_BUDGET = 1
MIN_REVENUE = 1
MIN_VOTE_COUNT = 1

# File paths
DATA_PATH = "data/"
RAW_DATASET_PATH = DATA_PATH + "movies_dataset.csv"
CLEAN_DATASET_PATH = DATA_PATH + "movies_dataset_clean.csv"