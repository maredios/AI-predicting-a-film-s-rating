import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import xgboost as xgb

from feature_engineering import preprocess_data


def train_model(clean_dataset_path="data/movies_dataset_clean.csv"):

    # Create folders if missing
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    log_path = "logs/training_log.txt"

    # -------------------------------------------------------------------------
    # 1. Load dataset
    # -------------------------------------------------------------------------
    print("Loading clean dataset...")
    df = pd.read_csv(clean_dataset_path)

    # -------------------------------------------------------------------------
    # 2. Feature engineering
    # -------------------------------------------------------------------------
    print("Preparing features...")
    X, y = preprocess_data(df)

    # Convert to DMatrix for xgboost.train()
    dtrain = xgb.DMatrix(X, label=y)

    # -------------------------------------------------------------------------
    # 3. XGBoost parameters (PRO Version)
    # -------------------------------------------------------------------------
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    num_rounds = 300  # Number of trees

    # Log file
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("=== TRAINING LOG ===\n")
        log.write(f"Dataset size: {X.shape}\n")
        log.write(f"Start time: {time.ctime()}\n")
        log.write(f"Number of boosting rounds: {num_rounds}\n\n")

    print(f"Training model ({num_rounds} trees)...")

    # -------------------------------------------------------------------------
    # 4. Manual training loop with TQDM
    # -------------------------------------------------------------------------
    booster = None
    start_time = time.time()

    # TQDM progress bar
    pbar = tqdm(
        total=num_rounds,
        desc="Training XGBoost",
        ncols=100,
        dynamic_ncols=True
    )

    tree_times = []

    for i in range(num_rounds):
        t0 = time.time()

        # Train 1 boosting iteration
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=1,
            xgb_model=booster
        )

        t1 = time.time()
        iteration_time = t1 - t0
        tree_times.append(iteration_time)

        # Update progress bar
        pbar.set_postfix({
            "tree_time": f"{iteration_time:.3f}s",
            "avg": f"{np.mean(tree_times):.3f}s",
            "eta": f"{(num_rounds - (i+1)) * np.mean(tree_times):.1f}s"
        })
        pbar.update(1)

    pbar.close()
    total_time = time.time() - start_time

    # -------------------------------------------------------------------------
    # 5. Save model
    # -------------------------------------------------------------------------
    model_path = "models/movie_xgb.json"
    booster.save_model(model_path)

    # -------------------------------------------------------------------------
    # 6. Save logs
    # -------------------------------------------------------------------------
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n=== TRAINING COMPLETED ===\n")
        log.write(f"Total time: {total_time:.2f} seconds\n")
        log.write(f"Average tree time: {np.mean(tree_times):.4f} seconds\n")
        log.write(f"Model saved to: {model_path}\n")

    print(f"\nTraining finished in {total_time:.2f} seconds.")
    print(f"Model saved ➜ {model_path}")
    print(f"Logs saved ➜ {log_path}")


if __name__ == "__main__":
    train_model()
