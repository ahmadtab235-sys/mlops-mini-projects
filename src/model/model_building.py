import numpy as np
import pandas as pd
import os
import pickle
from sklearn.linear_model import LogisticRegression
import yaml
import logging

# --- Logger setup ---
def setup_logger(name: str, log_file: str, level=logging.DEBUG) -> logging.Logger:
    """Set up a logger with console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

# --- Utility functions ---

def load_data(features_path: str) -> pd.DataFrame:
    """Load feature engineered training data"""
    train_data = pd.read_csv(os.path.join(features_path, "train_tfidf.csv"))
    return train_data

def processed_data(train_data: pd.DataFrame):
    """Split features and labels"""
    x_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    return x_train, y_train

def build_model(x_train, y_train):
    """Train Logistic Regression Model"""
    clf = LogisticRegression(C=1,solver='liblinear',penalty='l2')
    clf.fit(x_train, y_train)
    return clf

def save_file(clf, filename: str = os.path.join("model", "model.pkl")):
    """Save trained model to pickle file"""
    os.makedirs("model", exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(clf, f)

# --- Main workflow ---
def main() -> None:
    logger = setup_logger("model_building", "errorModelBuilding.log")
    logger.info("Starting model building...")

    train_data = load_data(os.path.join("data", "interim"))
    x_train, y_train = processed_data(train_data)

    clf = build_model(
        x_train,
        y_train
    )

    save_file(clf)
    logger.info("Model building completed successfully. Model saved in model/model.pkl")

if __name__ == "__main__":
    main()
