import mlflow
import dagshub

mlflow.set_tracking_uri('https://dagshub.com/ahmadtab235/mlops-mini-projects.mlflow')
dagshub.init(repo_owner='ahmadtab235', repo_name='mlops-mini-projects', mlflow=True)


with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)