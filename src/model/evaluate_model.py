import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from yaml import safe_load
from sklearn.pipeline import Pipeline
import joblib
import mlflow
import dagshub
import mlflow.sklearn
from mlflow.models import infer_signature
from src.model.train_model_rf import RandomForestRegressor
from mlflow.sklearn import log_model
from dotenv import load_dotenv
load_dotenv()
from src.logger import setup_logger
logger = setup_logger(name="evaluate_model")


tracking_uri=os.getenv("MLFLOW_TRACKING_URI")

if __name__ == "__main__":
   # dagshub and mlflow initialization

   dagshub.init(repo_owner='Rutu-RD', repo_name='flight_prediction', mlflow=True)
   try:
      mlflow.set_tracking_uri(tracking_uri)
   except Exception as e:
      logger.error("Failed to set MLflow tracking URI")
      raise e

   mlflow.set_experiment("Evaluation xgb and rf models")

   #getting x_val and y_val from splitted data

   x_val = pd.read_csv(os.path.join("data", "splitted_data", "X_val.csv"))
   y_val = pd.read_csv(os.path.join("data", "splitted_data", "y_val.csv")) 
   with mlflow.start_run(run_name="RF_Model_Evaluation"):
        logger.info("model evaluation starting")
        if isinstance(y_val, pd.DataFrame) and y_val.shape[1] == 1:
            y_val = y_val.iloc[:, 0]

        # load the model

        model_pipeline: Pipeline = joblib.load(os.path.join("models", "random_forest_model.pkl"))
        logger.info("random forest model loaded successfully")

        #log parameters for model

        with open("params.yaml") as f:
            params = safe_load(f)

        n_estimators = params['model']['random_forest']['n_estimators']
        max_depth = params['model']['random_forest']['max_depth']
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("n_estimators", n_estimators)
        logger.info(f"Logging RF model parameters: n_estimators={n_estimators}, max_depth={max_depth}")

        #log model_signature

        signature=infer_signature(x_val.head(10),model_pipeline.predict(x_val.head(10)))
        mlflow.sklearn.log_model(
        sk_model=model_pipeline,
        artifact_path="model_pipeline",
        signature=signature)
        logger.info("RF model signature logged to mlflow")
        #predicting on validation data
        y_pred = model_pipeline.predict(x_val)
        logger.info("prediction on validation data completed")
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        # logging metrics to mlflow
        logger.info(f"Mean Absolute Error: {mae}")
        logger.info(f"Mean Squared Error: {mse}")
        logger.info(f"R^2 Score: {r2}")
        
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("R2_Score", r2)
        logger.info("RF model evaluation metrics logged to mlflow")
        logger.info("RF model evaluation completed")
   


   with mlflow.start_run(run_name="XGB_Model_Evaluation"):
       logger.info(" xgboost model evaluation starting")
       if isinstance(y_val, pd.DataFrame) and y_val.shape[1] == 1:
            y_val = y_val.iloc[:, 0]
       # load the model

       model_pipeline: Pipeline = joblib.load(os.path.join("models", "xgboost_model.pkl"))
       logger.info("xgboost model loaded successfully")
        #log parameters for model
       try: 
           with open("params.yaml") as f:
               params = safe_load(f)
       except FileNotFoundError as e:
           logger.error("File not found: params.yaml")
           raise e
       
       n_estimators = params['model']['xgboost']['n_estimators']
       max_depth = params['model']['xgboost']['max_depth']
       learning_rate = params['model']['xgboost']['learning_rate']
       mlflow.log_param("max_depth", max_depth)
       mlflow.log_param("n_estimators", n_estimators)
       mlflow.log_param("learning_rate", learning_rate)
       logger.info(f"Logging XGB model parameters: n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}")
       # adding model signature
       signature=infer_signature(x_val.head(10),model_pipeline.predict(x_val.head(10)))
       mlflow.sklearn.log_model(sk_model=model_pipeline,artifact_path="model_pipeline",signature=signature)
       logger.info("XGB model signature logged to mlflow")
       #predicting on validation data

       y_pred = model_pipeline.predict(x_val)

       mae = mean_absolute_error(y_val, y_pred)
       mse = mean_squared_error(y_val, y_pred)
       r2 = r2_score(y_val, y_pred)
       logger.info(f"Mean Absolute Error: {mae}")
       logger.info(f"Mean Squared Error: {mse}")
       logger.info(f"R^2 Score: {r2}")
       
       mlflow.log_metric("MAE", mae)
       mlflow.log_metric("MSE", mse)
       mlflow.log_metric("R2_Score", r2)   
       logger.info("XGB model evaluation metrics logged to mlflow")
      
      
       logger.info("xgboost model evaluation completed")