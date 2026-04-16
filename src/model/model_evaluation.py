import pandas as pd
import numpy as np
import json
import pickle
import logging  
import os
import dagshub
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score



mlflow.set_tracking_uri('https://dagshub.com/ahmadtab235/mlops-mini-projects.mlflow')
dagshub.init(repo_owner='ahmadtab235', repo_name='mlops-mini-projects', mlflow=True)


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
def load_model(model_path: str):
    """Load trained model from pickle file"""
    clf = pickle.load(open(model_path, "rb"))
    return clf

def load_test_data(features_path: str) -> pd.DataFrame:
    """Load feature engineered test data"""
    test_data = pd.read_csv(os.path.join(features_path, "test_tfidf.csv"))
    return test_data

def prepare_test_data(test_data: pd.DataFrame):
    """Split features and labels"""
    x_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    return x_test, y_test

def evaluate_model(clf, x_test, y_test):
    """Evaluate model and compute metrics"""
    y_pred = clf.predict(x_test)
    y_pred_proba = clf.predict_proba(x_test)[:, 1]

    metrics_dict = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred)
    }
    return metrics_dict

def save_metrics(metrics_dict: dict, filename: str = os.path.join("reports", "metrics.json")):
    """Save evaluation metrics to JSON file"""
    os.makedirs("reports", exist_ok=True)
    with open(filename, "w") as file:
        json.dump(metrics_dict, file, indent=4)

# --- Main workflow ---
def main() -> None:
    mlflow.set_experiment("dvc-pipeline")
    with mlflow.start_run(run_name="Model Evaluation"):
        logger = setup_logger("model_evaluation", "errorModelEvaluation.log")
        logger.info("Starting model evaluation...")

        clf = load_model(os.path.join("model", "model.pkl"))
        test_data = load_test_data(os.path.join("data", "interim"))
        x_test, y_test = prepare_test_data(test_data)

        metrics_dict = evaluate_model(clf, x_test, y_test)
        save_metrics(metrics_dict)

        # --- Log parameters ---
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_param("num_features", x_test.shape[1])

        # --- Log metrics ---
        for key, value in metrics_dict.items():
            mlflow.log_metric(key, value)

        # --- Log model ---
        mlflow.sklearn.log_model(clf, artifact_path="model")
        mlflow.log_artifact('reports/metrics.json')

        logger.info("Model evaluation completed successfully. Metrics saved to reports/metrics.json")

if __name__ == "__main__":
    main()
