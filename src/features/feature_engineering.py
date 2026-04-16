import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml

# --- Logging setup ---
logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("error.logFeature")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# --- Utility functions ---
def load_params(params_path: str) -> int:
    """Load max_features from params.yaml"""
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params["feature_engineering"]["max_features"]

def load_data(processed_path: str):
    """Fetch the processed train and test data"""
    train_data = pd.read_csv(os.path.join(processed_path, "train_processed.csv"))
    test_data = pd.read_csv(os.path.join(processed_path, "test_processed.csv"))
    logger.info("Processed train and test data loaded successfully.")
    return train_data, test_data

def processed_data(train_data: pd.DataFrame, test_data: pd.DataFrame):
    """Prepare features and labels"""
    train_data.fillna("", inplace=True)
    test_data.fillna("", inplace=True)

    x_train = train_data["content"].values
    y_train = train_data["sentiment"].values

    x_test = test_data["content"].values
    y_test = test_data["sentiment"].values

    return x_train, y_train, x_test, y_test

def feature_engineering(x_train, y_train, x_test, y_test, max_features: int):
    """Apply TfidfVectorizer feature engineering"""
    vectorizer =TfidfVectorizer(max_features=max_features)

    x_train_bow = vectorizer.fit_transform(x_train)
    x_test_bow = vectorizer.transform(x_test)

    train_df = pd.DataFrame(x_train_bow.toarray())
    train_df["label"] = y_train

    test_df = pd.DataFrame(x_test_bow.toarray())
    test_df["label"] = y_test

    logger.info("Feature engineering completed successfully.")
    return train_df, test_df

def save_data(data_path: str, train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Save engineered features into data/interim"""
    os.makedirs(data_path, exist_ok=True)
    train_df.to_csv(os.path.join(data_path, "train_tfidf.csv"), index=False)
    test_df.to_csv(os.path.join(data_path, "test_tfidf.csv"), index=False)
    logger.info("Feature engineered data saved successfully in data/interim/")

# --- Main workflow ---
def main() -> None:
    logger.info("Starting feature engineering...")
    max_features = load_params("params.yaml")
    train_data, test_data = load_data(os.path.join("data", "processed"))
    x_train, y_train, x_test, y_test = processed_data(train_data, test_data)
    train_df, test_df = feature_engineering(x_train, y_train, x_test, y_test, max_features)
    save_data(os.path.join("data", "interim"), train_df, test_df)
    logger.info("Feature engineering stage completed successfully.")

if __name__ == "__main__":
    main()
