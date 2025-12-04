# train_model.py
import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from feature_engineering import (
    preprocess_data,
    save_encoders,
)


def train_model(clean_dataset_path="data/movies_dataset_clean.csv"):

    # ------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    log_path = "logs/training_log.txt"
    encoders_path = "models/encoders.pkl"
    model_path = "models/movie_xgb.json"
    curve_path = "plots/learning_curve.png"

    # ------------------------------------------------------------
    # Load clean dataset
    # ------------------------------------------------------------
    print("Loading clean dataset...")
    df = pd.read_csv(clean_dataset_path)

    # ------------------------------------------------------------
    # Feature engineering + encoders
    # ------------------------------------------------------------
    print("Preparing encoders and features...")

    # preprocess_data builds encoders + returns X,y
    X, y, encoders = preprocess_data(df, encoders_path)

    # Save encoders to file
    save_encoders(encoders, encoders_path)

    # ------------------------------------------------------------
    # Train/test split
    # ------------------------------------------------------------
    print("Splitting train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # ------------------------------------------------------------
    # XGBoost parameters
    # ------------------------------------------------------------
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    num_rounds = 300
    print(f"Training model ({num_rounds} trees)...")

    # ------------------------------------------------------------
    # Training loop + RMSE tracking
    # ------------------------------------------------------------
    booster = None
    start = time.time()

    train_rmse_list = []
    test_rmse_list = []

    pbar = tqdm(total=num_rounds, desc="Training XGBoost", ncols=100, dynamic_ncols=True)

    for i in range(num_rounds):

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=1,
            xgb_model=booster
        )

        # Predictions for RMSE
        train_pred = booster.predict(dtrain)
        test_pred = booster.predict(dtest)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse  = np.sqrt(mean_squared_error(y_test, test_pred))


        train_rmse_list.append(train_rmse)
        test_rmse_list.append(test_rmse)

        pbar.set_postfix({
            "train_RMSE": f"{train_rmse:.3f}",
            "test_RMSE": f"{test_rmse:.3f}",
        })
        pbar.update(1)

    pbar.close()
    total_time = time.time() - start

    # ------------------------------------------------------------
    # Save model & logs
    # ------------------------------------------------------------
    booster.save_model(model_path)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== TRAINING LOG ===\n")
        f.write(f"Dataset size: {X.shape}\n")
        f.write(f"Training duration: {total_time:.2f} sec\n")
        f.write(f"Final Train RMSE: {train_rmse_list[-1]:.4f}\n")
        f.write(f"Final Test  RMSE: {test_rmse_list[-1]:.4f}\n")
        f.write(f"Model saved at: {model_path}\n")
        f.write(f"Encoders saved at: {encoders_path}\n")

    print(f"\nTraining finished in {total_time:.2f}s")
    print(f"Final Train RMSE: {train_rmse_list[-1]:.4f}")
    print(f"Final Test  RMSE: {test_rmse_list[-1]:.4f}")
    print(f"Model saved → {model_path}")
    print(f"Encoders saved → {encoders_path}")
    print(f"Log saved → {log_path}")

    # ------------------------------------------------------------
    # Learning Curve plot
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(train_rmse_list, label="Train RMSE")
    plt.plot(test_rmse_list, label="Test RMSE")
    plt.xlabel("Boosting Rounds (Trees)")
    plt.ylabel("RMSE")
    plt.title("Learning Curve (RMSE)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path)
    plt.close()

    print(f"Learning curve saved → {curve_path}")


if __name__ == "__main__":
    train_model()
