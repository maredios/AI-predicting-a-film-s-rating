import os
import subprocess

def run():
    print("Cleaning dataset...")
    subprocess.run(["python", "data_cleaning.py"])

    print("Training model...")
    subprocess.run(["python", "train_model.py"])

    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    run()
