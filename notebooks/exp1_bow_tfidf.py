import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords 
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import mlflow.sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import dagshub

mlflow.set_tracking_uri('https://dagshub.com/ahmadtab235/mlops-mini-projects.mlflow')
dagshub.init(repo_owner='ahmadtab235', repo_name='mlops-mini-projects', mlflow=True)

df = pd.read_csv(r"C:\Users\TABISH AHMAD\Documents\DataSets\tweet_emotions.csv").drop(columns=['tweet_id'])

stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

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

def normalize_text(df: pd.DataFrame, colname: str) -> pd.DataFrame:
    df[colname] = df[colname].astype(str)
    df[colname] = df[colname].apply(lower_case)
    df[colname] = df[colname].apply(removing_urls)
    df[colname] = df[colname].apply(removing_punctuations)
    df[colname] = df[colname].apply(remove_stop_words)
    df[colname] = df[colname].apply(lemmatization)
    return df

df = normalize_text(df, "content")
df = df[df['sentiment'].isin(['happiness', 'sadness'])]

mlflow.set_experiment("Bow vs Tfidf")

vectorizer = {
    'Bow': CountVectorizer(),
    'TF-IDF': TfidfVectorizer()
}
algorithms = {
    'LogisticRegression': LogisticRegression(),
    'MultinomialNB': MultinomialNB(),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

with mlflow.start_run(run_name="All-Experiment") as parent_run:
    for algo_name, algo in algorithms.items():   # use 'algo' instead of 'algorithms'
        for vec_name, vec in vectorizer.items(): # use 'vec' instead of 'vectorizer'
            with mlflow.start_run(run_name=f"{algo_name} with {vec_name}", nested=True) as child_run:
                X = vec.fit_transform(df['content'])

                # Encode labels only for XGBoost
                if algo_name == 'XGBoost':
                    le = LabelEncoder()
                    y = le.fit_transform(df['sentiment'])
                else:
                    y = df['sentiment']

                x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

                mlflow.log_param("vectorizer", vec_name)
                mlflow.log_param("algorithm", algo_name)
                mlflow.log_param("test_size", 0.20)

                model = algo
                model.fit(x_train, y_train)

                # log model parameters
                if algo_name == 'LogisticRegression':
                    mlflow.log_param("C", model.C)
                elif algo_name == 'MultinomialNB':
                    mlflow.log_param("alpha", model.alpha)
                elif algo_name == 'XGBoost':
                    mlflow.log_param("n_estimators", model.n_estimators)
                    mlflow.log_param("learning_rate", model.learning_rate)
                elif algo_name == 'RandomForest':
                    mlflow.log_param("n_estimators", model.n_estimators)
                    mlflow.log_param("max_depth", model.max_depth)

                # evaluation
                y_pred = model.predict(x_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')

                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)

                mlflow.sklearn.log_model(model, artifact_path="model")

                print(f"Algorithms : {algo_name}, Feature Engineering : {vec_name}")
                print(f"Accuracy : {accuracy}")
                print(f"Precision: {precision}")
                print(f"Recall : {recall}")
                print(f"F1_score : {f1}")
