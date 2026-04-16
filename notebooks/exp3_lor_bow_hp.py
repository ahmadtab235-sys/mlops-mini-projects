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
from sklearn.model_selection import GridSearchCV

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

vectorizer=CountVectorizer()
x=vectorizer.fit_transform(df['content'])
y=df['sentiment']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)


mlflow.set_experiment("LoR Hyperparameter Tuning")

param_grid={
    'C':[0.1,1,10],
    'penalty':['l1','l2'],
    'solver':['liblinear']
}


with mlflow.start_run():
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(x_train, y_train)

    for params, mean_score, std_score in zip(
        grid_search.cv_results_['params'],
        grid_search.cv_results_['mean_test_score'],
        grid_search.cv_results_['std_test_score']
    ):
        with mlflow.start_run(run_name=f"LR with params: {params}", nested=True):
            model = LogisticRegression(**params)
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            # log params correctly
            for key, value in params.items():
                mlflow.log_param(key, value)

            mlflow.log_metric("mean_cv_score", mean_score)
            mlflow.log_metric("std_cv_score", std_score)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"Mean CV score : {mean_score}, STD CV Score: {std_score}")
            print(f"Accuracy : {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall : {recall}")
            print(f"F1_score : {f1}")

best_params = grid_search.best_params_
best_score = grid_search.best_score_

for key, value in best_params.items():
    mlflow.log_param(key, value)
mlflow.log_metric("best_f1_score", best_score)

print(f"Best Params : {best_params}")
print(f"Best f1 score: {best_score}")

