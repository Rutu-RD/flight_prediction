import pandas as pd
import numpy as np
import os
import logging
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from yaml import safe_load
from src.model.pipeline import build_preprocessor 
import joblib
import mlflow
import dagshub
import mlflow.sklearn
from mlflow.models import infer_signature
from dotenv import load_dotenv
load_dotenv()
import warnings
warnings.filterwarnings("ignore")

from src.logger import setup_logger
logger = setup_logger(name="train_model_xgboost")

def check_credentials():
    mlflow_username=os.getenv("MLFLOW_TRACKING_USERNAME")
    mlflow_password=os.getenv("MLFLOW_TRACKING_PASSWORD")
    tracking_uri=os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_username is None or mlflow_password is None:
        logger.error("MLflow tracking credentials are not set in environment variables.")
        return False
    return tracking_uri

if __name__=="__main__":
    logger.info("train_xgboost_tracking")
    tracking_uri = check_credentials()
    
    
    x_train = pd.read_csv(os.path.join("data", "splitted_data", "X_train.csv"))
    y_train = pd.read_csv(os.path.join("data", "splitted_data", "y_train.csv"))

    if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
        y_train = y_train.iloc[:, 0]

    with open("params.yaml") as f:
        params = safe_load(f)
    n_estimators = params['model']['xgboost']['n_estimators']
    max_depth = params['model']['xgboost']['max_depth']
    learning_rate = params['model']['xgboost']['learning_rate']
    random_state = params['model']['xgboost']['random_state']
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("train_model_xgboost_tracking")
    with mlflow.start_run(run_name="xgboost_model_tracking"):
        xgb = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state
            )
        model_pipeline = Pipeline(steps=[
            ('preprocessor', build_preprocessor()),
            ('model', xgb)
        ])
        logger.info(" xgb model pipeline created")
        model_pipeline.fit(x_train, y_train)
        mlflow.log_param("n_estimators",n_estimators)
        mlflow.log_param("max_depth",max_depth)
        mlflow.log_param("learning_rate",learning_rate)
        mlflow.log_param("random_state",random_state)
        logger.info("xgboost parameters logged to mlflow")
        signature=infer_signature(x_train.head(10),model_pipeline.predict(x_train.head(10)))
        mlflow.sklearn.log_model(sk_model=model_pipeline,artifact_path="model_pipeline",signature=signature)
        

        logger.info("xgboost model training completed")
        os.makedirs("models", exist_ok=True)
        joblib.dump(model_pipeline, os.path.join("models", "xgboost_model.pkl"))
        logger.info("xgboost model saved successfully")