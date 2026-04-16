import numpy as np
import pandas as pd
import os
import re
import nltk
import logging
from nltk.corpus import stopwords 
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# --- Logging setup ---
logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("error.logPre")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# --- Config ---
TEXT_COLUMN = "content"
RAW_DATA_PATH = os.path.join("data", "raw")
PROCESSED_DATA_PATH = os.path.join("data", "processed")

def load_raw_data(raw_path: str):
    train_data = pd.read_csv(os.path.join(raw_path, "train.csv"))
    test_data = pd.read_csv(os.path.join(raw_path, "test.csv"))
    logger.info("Train and test data loaded from data/raw/")
    return train_data, test_data

# Download required NLTK resources
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("punkt")

# Initialize tools once
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")

def lemmatization(text: str) -> str:
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

def remove_stop_words(text: str) -> str:
    tokens = text.split()
    tokens = [word for word in tokens if word.lower() not in stop_words]
    return " ".join(tokens)

def lower_case(text: str) -> str:
    tokens = text.split()
    tokens = [word.lower() for word in tokens]
    return " ".join(tokens)

def removing_punctuations(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def removing_urls(text: str) -> str:
    urls_pattern = re.compile(r"https?://\S+|www\.\S+")
    return urls_pattern.sub(r"", text)

def remove_small_sentences(df: pd.DataFrame, colname: str) -> pd.DataFrame:
    df.loc[df[colname].str.split().str.len() < 3, colname] = np.nan
    return df

def normalize_text(df: pd.DataFrame, colname: str) -> pd.DataFrame:
    df[colname] = df[colname].astype(str)
    df[colname] = df[colname].apply(lower_case)
    df[colname] = df[colname].apply(removing_urls)
    df[colname] = df[colname].apply(removing_punctuations)
    df[colname] = df[colname].apply(remove_stop_words)
    df[colname] = df[colname].apply(lemmatization)
    return df

def processed_data(train_data: pd.DataFrame, test_data: pd.DataFrame):
    logger.info("Normalizing train data...")
    train_processed_data = normalize_text(train_data, TEXT_COLUMN)
    train_processed_data = remove_small_sentences(train_processed_data, TEXT_COLUMN)

    logger.info("Normalizing test data...")
    test_processed_data = normalize_text(test_data, TEXT_COLUMN)
    test_processed_data = remove_small_sentences(test_processed_data, TEXT_COLUMN)

    return train_processed_data, test_processed_data

def save_data(data_path: str, train_processed_data: pd.DataFrame, test_processed_data: pd.DataFrame):
    os.makedirs(data_path, exist_ok=True)
    train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
    test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
    logger.info("Preprocessed train/test data saved in data/processed/")

def main() -> None:
    logger.info("Starting data preprocessing...")
    train_data, test_data = load_raw_data(RAW_DATA_PATH)
    train_processed, test_processed = processed_data(train_data, test_data)
    save_data(PROCESSED_DATA_PATH, train_processed, test_processed)
    logger.info("Data preprocessing completed successfully.")

if __name__ == "__main__":
    main()
