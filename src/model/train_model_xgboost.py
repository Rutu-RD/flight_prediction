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
from dotenv import load_dotenv
load_dotenv()
tracking_uri=os.getenv("MLFLOW_TRACKING_URI")
logging.basicConfig(level=logging.INFO)
console = logging.StreamHandler()
logger = logging.getLogger(__name__)
logger.addHandler(console)

if __name__=="__main__":
    logger.info("flight prediction train_xgboost_tracking")
    
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
        model_pipeline.fit(x_train, y_train)
        mlflow.log_param("n_estimators",n_estimators)
        mlflow.log_param("max_depth",max_depth)
        mlflow.log_param("learning_rate",learning_rate)
        mlflow.log_param("random_state",random_state)

        logger.info("xgboost model training completed")
        os.makedirs("models", exist_ok=True)
        joblib.dump(model_pipeline, os.path.join("models", "xgboost_model.pkl"))
        logger.info("xgboost model saved successfully")