import os
import logging
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from yaml import safe_load, safe_dump
from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn
import dagshub
import mlflow.sklearn
from mlflow.models import infer_signature
from src.model.pipeline import build_preprocessor
from dotenv import load_dotenv
load_dotenv()

def check_credentials():
    mlflow_username=os.getenv("MLFLOW_TRACKING_USERNAME")
    mlflow_password=os.getenv("MLFLOW_TRACKING_PASSWORD")
    tracking_uri=os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_username is None or mlflow_password is None:
        logger.error("MLflow tracking credentials are not set in environment variables.")
        return False
    return tracking_uri
import warnings
warnings.filterwarnings("ignore")
from src.logger import setup_logger 
logger = setup_logger(name="hyperparameter_tuning_eval")

if __name__ == "__main__":
    
    tracking_uri = check_credentials()
     # Set MLflow tracking URI
    try:
        mlflow.set_tracking_uri(tracking_uri)
    except Exception as e:
        logger.error("Failed to set MLflow tracking URI")
        raise e
    mlflow.set_experiment("hyperparameter_tuning RF and xgboost")

    # Load data (train + val)
    # getting x_train, y_train, x_val, y_val from splitted data
    x_train = pd.read_csv(os.path.join("data", "splitted_data", "X_train.csv"))
    y_train = pd.read_csv(os.path.join("data", "splitted_data", "y_train.csv"))
    x_val = pd.read_csv(os.path.join("data", "splitted_data", "X_val.csv"))
    y_val = pd.read_csv(os.path.join("data", "splitted_data", "y_val.csv"))
    logger.info("train and validate splitted data loaded successfully")
    if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
        y_train = y_train.iloc[:, 0]
    if isinstance(y_val, pd.DataFrame) and y_val.shape[1] == 1:
        y_val = y_val.iloc[:, 0]

    # Read current params (base config)
    try:
         logger.info("reading base params from params.yaml file")
         with open("params.yaml") as f:
             params = safe_load(f)
    except FileNotFoundError as e:
         logger.error("File not found: params.yaml")
         raise e
    

    
    # Hyperparameter training parameters for random forest
   
    n_estimators_list = params["hyperparameter_models"]["random_forest"]["n_estimators"]
    depth = params["hyperparameter_models"]["random_forest"]["max_depth"]
    

    #  Parent run for RF hyperparameter search
    with mlflow.start_run(run_name="rf_hyperparameter_tuning") as parent_run:
        logger.info(f"Parent run id: {parent_run.info.run_id}")
        for n in n_estimators_list:
            for d in depth:
                run_name = f"rf_n_estimators{n}_depth{d}"
                with mlflow.start_run(run_name=run_name, nested=True) as child_run:
                    logger.info(f"Starting child run: {run_name}")
                    rf_hp = RandomForestRegressor(
                        n_estimators=n,
                        max_depth=d,
                        random_state=42,
                        n_jobs=-1
                    )

                    model_pipeline = Pipeline(steps=[
                        ("preprocessor", build_preprocessor()),
                        ("model", rf_hp),
                    ])

                    # Fit pipeline
                    logger.info("Fitting model_pipeline...")
                    model_pipeline.fit(x_train, y_train)

                    # Log params
                    mlflow.log_param("n_estimators", n)
                    mlflow.log_param("max_depth", d)
                    mlflow.log_param("model_type", "RandomForestRegressor")
                    logger.info(f"Logged parameters: n_estimators={n}, max_depth={d}")

                    #log model
                    signature=infer_signature(x_train.head(10),model_pipeline.predict(x_train.head(10)))
                    mlflow.sklearn.log_model(sk_model=model_pipeline,artifact_path="model_pipeline",signature=signature)
                    logger.info("Model logged to mlflow")
                    # Evaluate on validation set
                    logger.info("Evaluating on validation set...")
                    y_pred = model_pipeline.predict(x_val)

                    mae = mean_absolute_error(y_val, y_pred)
                    mse = mean_squared_error(y_val, y_pred)
                    r2 = r2_score(y_val, y_pred)

                    logger.info(f"[{run_name}] MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f}")

                    # Log metrics to mlflow
                    mlflow.log_metric("MAE", mae)
                    mlflow.log_metric("MSE", mse)
                    mlflow.log_metric("R2_Score", r2)
                    logger.info(f"Metrics logged for run: {run_name}")

                    




    # Hyperparameter training for xgboost
    with mlflow.start_run(run_name="xgb_hyperparameter_tuning") as parent_run:
        logger.info("starting parent run for xgboost")
        # load hyperparameter values from params.yaml
        try: 
             logger.info("reading hyperparameter tuning params from params.yaml file")
             with open("params.yaml") as f:
                 params = safe_load(f)
        except FileNotFoundError as e:
             logger.error("File not found: params.yaml")
             raise e
        
        learning_rates = params["hyperparameter_models"]["xgboost"]["learning_rate"]
        n_estimators_list = params["hyperparameter_models"]["xgboost"]["n_estimators"]
        depths = params["hyperparameter_models"]["xgboost"]["max_depth"]
        for lr in learning_rates:
            for n in n_estimators_list:
                for d in depths:
                    run_name=f"xgboost {lr} lr n{n} {d}d(depth)"
                    with mlflow.start_run(run_name=run_name,nested=True) as child_run:
                        logger.info(f"starting child run {run_name}")
                        xgb_hp = XGBRegressor(
                            learning_rate=lr,
                            n_estimators=n,
                            max_depth=d,
                            random_state=42
                        )
                        model_pipeline=Pipeline(steps=[
                            ("processor",build_preprocessor()),
                            ("model",xgb_hp)])
                        # Fit pipeline

                        logger.info("fitting xgboost model pipeline")
                        model_pipeline.fit(x_train,y_train)

                        #log params to mlfloww

                        mlflow.log_param("learning_rate",lr)
                        mlflow.log_param("n_estimators",n)
                        mlflow.log_param("max_depth",d)
                        mlflow.log_param("model_type", "XGBRegressor")
                        logger.info(f"logged xgboost parameters: learning_rate={lr}, n_estimators={n}, max_depth={d}")

                        #log model to mlflow
                        signature=infer_signature(x_train.head(10),model_pipeline.predict(x_train.head(10)))
                        mlflow.sklearn.log_model(sk_model=model_pipeline,artifact_path="model_pipeline",signature=signature)
                        logger.info("xgboost model f{run_name} logged to mlflow".format(run_name=run_name))
                        logger.info("evaluating on validation set")
                        y_pred=model_pipeline.predict(x_val)
                        #log metrics to mlflow
                        mae=mean_absolute_error(y_val,y_pred)
                        mse=mean_squared_error(y_val,y_pred)
                        r2=r2_score(y_val,y_pred)
                        logger.info(f"[{run_name}] MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f}")
                        mlflow.log_metric("MAE",mae)
                        mlflow.log_metric("MSE",mse)
                        mlflow.log_metric("R2_Score",r2)
                        logger.info(f"xgboost metrics logged for run: {run_name}")
                        
                        # mlflow.log_param("model_type","XGBRegressor")
                        # mlflow.sklearn.log_model(
                        #     sk_model=xgb_hp,
                        #     artifact_path="model_pipeline",
                        # )
