import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from yaml import safe_load
from sklearn.pipeline import Pipeline
import joblib
logging.basicConfig(level=logging.INFO)
console = logging.StreamHandler()
logger = logging.getLogger(__name__)
logger.addHandler(console)
import mlflow
import dagshub
import mlflow.sklearn


if __name__ == "__main__":
   dagshub.init(repo_owner='Rutu-RD', repo_name='dagshub_flight_prediction', mlflow=True)
   mlflow.set_tracking_uri("https://dagshub.com/Rutu-RD/dagshub_flight_prediction.mlflow")
   dagshub.init(repo_owner='Rutu-RD', repo_name='dagshub_flight_prediction', mlflow=True)
   mlflow.set_experiment("flight_price_random_forest_experiment")
   with mlflow.start_run(run_name="random_forest_model_evaluation"):
        logger.info("model evaluation starting")

        x_test = pd.read_csv(os.path.join("data", "splitted_data", "X_val.csv"))
        y_test = pd.read_csv(os.path.join("data", "splitted_data", "y_val.csv"))

        if isinstance(y_test, pd.DataFrame) and y_test.shape[1] == 1:
            y_test = y_test.iloc[:, 0]

        model_pipeline: Pipeline = joblib.load(os.path.join("models", "random_forest_model.pkl"))
        logger.info("random forest model loaded successfully")
        y_pred = model_pipeline.predict(x_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logger.info(f"Mean Absolute Error: {mae}")
        logger.info(f"Mean Squared Error: {mse}")
        logger.info(f"R^2 Score: {r2}")
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("R2_Score", r2)
        mlflow.sklearn.log_model(model_pipeline, "random_forest_model")
        
        with open("params.yaml") as f:
            params = safe_load(f)
        n = params['model']['random_forest']['n_estimators']
        max_depth = params['model']['random_forest']['max_depth']
        mlflow.log_param("max_depth", max_depth)

        mlflow.log_param("n_estimators", n)
        logger.info("RF model evaluation completed")
#    with mlflow.start_run(run_name="xgboost_model_evaluation"):


 #       logger.info(" xgboost model evaluation starting")
 #       model_pipeline: Pipeline = joblib.load(os.path.join("models", "xgboost_model.pkl"))
 #       logger.info("xgboost model loaded successfully")
 #       y_pred = model_pipeline.predict(x_test)
 #       mae = mean_absolute_error(y_test, y_pred)
 #       mse = mean_squared_error(y_test, y_pred)
  #      r2 = r2_score(y_test, y_pred)
 #       logger.info(f"Mean Absolute Error: {mae}")
 #       logger.info(f"Mean Squared Error: {mse}")
 #       logger.info(f"R^2 Score: {r2}")
  #      logger.info("xgboost model evaluation completed")
  #      mlflow.log_metric("MAE", mae)
   #     mlflow.log_metric("MSE", mse)
   #     mlflow.log_metric("R2_Score", r2)   
