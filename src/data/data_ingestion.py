import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

# --- Logging setup ---
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("errordata_ingestions.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# --- Utility functions ---
def load_params(params_path: str) -> float:
    """Load test_size from params.yaml"""
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params["data_ingestion"]["test_size"]

def read_data(file_path: str) -> pd.DataFrame:
    """Read CSV file"""
    df = pd.read_csv(file_path)
    return df

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unnecessary column and filter sentiments"""
    df.drop(columns=["tweet_id"], inplace=True)
    final_df = df[df["sentiment"].isin(["happiness", "sadness"])].copy()
    final_df["sentiment"] = final_df["sentiment"].map({"happiness": 1, "sadness": 0}).astype(int)
    return final_df

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """Save train/test CSVs into data/raw"""
    raw_path = os.path.join("data", "raw")
    os.makedirs(raw_path, exist_ok=True)
    train_data.to_csv(os.path.join(raw_path, "train.csv"), index=False)
    test_data.to_csv(os.path.join(raw_path, "test.csv"), index=False)

# --- Main workflow ---
def main() -> None:
    logger.info("Starting data ingestion...")
    test_size = load_params("params.yaml")

    # Adjust this path to wherever your original dataset lives
    df = read_data(r"C:\Users\TABISH AHMAD\Documents\DataSets\tweet_emotions.csv")

    final_df = process_data(df)
    train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

    save_data(train_data, test_data)
    logger.info("Data ingestion completed successfully. Train/Test saved in data/raw/")

if __name__ == "__main__":
    main()
