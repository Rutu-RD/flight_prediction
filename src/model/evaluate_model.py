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
from sklearn.model_selection import KFold,cross_val_score
from mlflow.sklearn import log_model
from dotenv import load_dotenv
load_dotenv()
from src.logger import setup_logger

logger = setup_logger(name="evaluate_model")
def check_credentials():
    mlflow_username=os.getenv("MLFLOW_TRACKING_USERNAME")
    mlflow_password=os.getenv("MLFLOW_TRACKING_PASSWORD")
    tracking_uri=os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_username is None or mlflow_password is None:
        logger.error("MLflow tracking credentials are not set in environment variables.")
        return False
    return tracking_uri



if __name__ == "__main__":
   # dagshub and mlflow initialization
   tracking_uri = check_credentials()
   
   try:
      mlflow.set_tracking_uri(tracking_uri)
   except Exception as e:
      logger.error("Failed to set MLflow tracking URI")
      raise e

   mlflow.set_experiment("Evaluation xgb and rf models")

   #getting x_val and y_val from splitted data
   x_train = pd.read_csv(os.path.join("data", "splitted_data", "X_train.csv")) 
   y_train = pd.read_csv(os.path.join("data", "splitted_data", "y_train.csv"))    
   x_val = pd.read_csv(os.path.join("data", "splitted_data", "X_val.csv"))
   y_val = pd.read_csv(os.path.join("data", "splitted_data", "y_val.csv"))
   x_test = pd.read_csv(os.path.join("data", "splitted_data", "X_test.csv"))
   y_test = pd.read_csv(os.path.join("data", "splitted_data", "y_test.csv"))
   if isinstance(y_test, pd.DataFrame) and y_test.shape[1] == 1:
            y_test = y_test.iloc[:, 0] 
   
   if isinstance(y_val, pd.DataFrame) and y_val.shape[1] == 1:
            y_val = y_val.iloc[:, 0]

   with mlflow.start_run(run_name="RF_Model_Evaluation"):
        logger.info("model evaluation starting")
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


        cv=KFold(n_splits=10, shuffle=True, random_state=42)
        mlflow.log_param("kfolds", cv.n_splits)

        scores=cross_val_score(model_pipeline, x_train, y_train, cv=cv, scoring='neg_mean_absolute_error')
        mean_score=np.mean(scores)
        logger.info(f"Cross-validated MAE on validation data: {-mean_score}")
        mlflow.log_metric("cv_neg_mae_on_validation_data", mean_score)

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
        mae_rf_val = mean_absolute_error(y_val, y_pred)
        mse_rf_val = mean_squared_error(y_val, y_pred)
        r2_rf_val = r2_score(y_val, y_pred)

        # logging metrics to mlflow
        logger.info(f"Mean Absolute Error on validation data: {mae_rf_val}")
        logger.info(f"Mean Squared Error on validation data: {mse_rf_val}")
        logger.info(f"R^2 Score on validation data: {r2_rf_val}")
        
        mlflow.log_metric("MAE_RF_VAL", mae_rf_val)
        mlflow.log_metric("MSE_RF_VAL", mse_rf_val)
        mlflow.log_metric("R2_RF_VAL", r2_rf_val)
        logger.info("RF model evaluation metrics logged to mlflow")
        logger.info("RF model evaluation completed")

        logger.info("prediction on test data starting ")
        
        y_pred_test = model_pipeline.predict(x_test)
        mae_rf_test = mean_absolute_error(y_test, y_pred_test)
        mse_rf_test = mean_squared_error(y_test, y_pred_test)
        r2_rf_test = r2_score(y_test, y_pred_test)
        logger.info(f"Mean Absolute Error on test data: {mae_rf_test}")
        logger.info(f"Mean Squared Error on test data: {mse_rf_test}")
        logger.info(f"R^2 Score on test data: {r2_rf_test}")
        mlflow.log_metric("MAE_RF_TEST", mae_rf_test)
        mlflow.log_metric("MSE_RF_TEST", mse_rf_test)
        mlflow.log_metric("R2_RF_TEST", r2_rf_test)
   


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

       cv=KFold(n_splits=10, shuffle=True,random_state=42)
       mlflow.log_param("kfolds", cv.n_splits)
       scores=cross_val_score(model_pipeline, x_train, y_train, cv=cv, scoring='neg_mean_absolute_error')

       mean_score=np.mean(scores)
       logger.info(f"Cross-validated MAE on validation data: {-mean_score}")
       mlflow.log_metric("cv_neg_mae_on_validation_data", mean_score)


       

       logger.info(f"Logging XGB model parameters: n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}")
       # adding model signature
       signature=infer_signature(x_val.head(10),model_pipeline.predict(x_val.head(10)))
       mlflow.sklearn.log_model(sk_model=model_pipeline,artifact_path="model_pipeline",signature=signature)
       logger.info("XGB model signature logged to mlflow")
       #predicting on validation data

       y_pred = model_pipeline.predict(x_val)

       mae_xgb_val = mean_absolute_error(y_val, y_pred)
       mse_xgb_val = mean_squared_error(y_val, y_pred)
       r2_xgb_val = r2_score(y_val, y_pred)
       logger.info(f"Mean Absolute Error on validation data: {mae_xgb_val}")
       logger.info(f"Mean Squared Error on validation data: {mse_xgb_val}")
       logger.info(f"R^2 Score on validation data: {r2_xgb_val}")
       
       mlflow.log_metric("MAE_XGB_VAL", mae_xgb_val)
       mlflow.log_metric("MSE_XGB_VAL", mse_xgb_val)
       mlflow.log_metric("R2_XGB_VAL", r2_xgb_val)

     
       y_pred_test = model_pipeline.predict(x_test)
       mae_xgb_test = mean_absolute_error(y_test, y_pred_test)
       mse_xgb_test = mean_squared_error(y_test, y_pred_test)
       r2_xgb_test = r2_score(y_test, y_pred_test)
       logger.info(f"Mean Absolute Error on test data: {mae_xgb_test}")
       logger.info(f"Mean Squared Error on test data: {mse_xgb_test}")
       logger.info(f"R^2 Score on test data: {r2_xgb_test}")
       mlflow.log_metric("MAE_XGB_TEST", mae_xgb_test)
       mlflow.log_metric("MSE_XGB_TEST", mse_xgb_test)
       mlflow.log_metric("R2_XGB_TEST", r2_xgb_test)

    



       logger.info("XGB model evaluation metrics logged to mlflow")
      
      
       logger.info("xgboost model evaluation completed")