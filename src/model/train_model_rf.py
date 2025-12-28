import pandas as pd
import numpy as np
import os
import logging
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from yaml import safe_load
from src.model.pipeline import build_preprocessor 
import joblib
import mlflow
import dagshub
import mlflow.sklearn
from mlflow.models import infer_signature
from dotenv import load_dotenv
load_dotenv()
tracking_uri=os.getenv("MLFLOW_TRACKING_URI")
logging.basicConfig(level=logging.INFO)
console = logging.StreamHandler()
logger = logging.getLogger(__name__)
logger.addHandler(console)

if __name__=="__main__":
   dagshub.init(repo_owner='Rutu-RD', repo_name='flight_prediction', mlflow=True)
   
   mlflow.set_tracking_uri(tracking_uri)
   
   mlflow.set_experiment("flight_price_train_model_rf_tracking")
   with mlflow.start_run(run_name="random_forest_model_training"):
        logger.info("training random forest regressor model")   
        x_train = pd.read_csv(os.path.join("data", "splitted_data", "X_train.csv"))
        y_train = pd.read_csv(os.path.join("data", "splitted_data", "y_train.csv"))

        if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
            y_train = y_train.iloc[:, 0]

        with open("params.yaml") as f:
            params = safe_load(f)
        n_estimators = params['model']['random_forest']['n_estimators']
        max_depth = params['model']['random_forest']['max_depth']
       
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

        model_pipeline = Pipeline(steps=[
            ('preprocessor', build_preprocessor()),
            ('model', rf)
        ])
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        model_pipeline.fit(x_train, y_train)
        logger.info("model training completed")
        # getting model signature and logging model to mlflow
        signature=infer_signature(x_train.head(5), model_pipeline.predict(x_train.head(5)))
        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            artifact_path="model_pipeline",
            signature=signature)
        logger.info("model logged to mlflow")
      
        #mlflow.sklearn.log_model(model_pipeline, "random_forest_model")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model_pipeline, os.path.join("models", "random_forest_model.pkl"))
        logger.info("model saved successfully")
