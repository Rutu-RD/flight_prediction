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

from src.model.pipeline import build_preprocessor
from dotenv import load_dotenv
load_dotenv()
tracking_uri=os.getenv("MLFLOW_TRACKING_URI")

logging.basicConfig(level=logging.INFO)
console = logging.StreamHandler()
logger = logging.getLogger(__name__)
logger.addHandler(console)
# getting x_train, y_train, x_val, y_val from splitted data
x_train = pd.read_csv(os.path.join("data", "splitted_data", "X_train.csv"))
y_train = pd.read_csv(os.path.join("data", "splitted_data", "y_train.csv"))
x_val = pd.read_csv(os.path.join("data", "splitted_data", "X_val.csv"))
y_val = pd.read_csv(os.path.join("data", "splitted_data", "y_val.csv"))

if __name__ == "__main__":
    # Init DagsHub + MLflow
    dagshub.init(repo_owner="Rutu-RD", repo_name="flight_prediction", mlflow=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("flight_price_hyperparameter_tuning_experiments")

    # Load data (train + val)
    

    if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
        y_train = y_train.iloc[:, 0]
    if isinstance(y_val, pd.DataFrame) and y_val.shape[1] == 1:
        y_val = y_val.iloc[:, 0]

    # Read current params (base config)
    with open("params.yaml") as f:
        params = safe_load(f)

    base_n = params["model"]["random_forest"]["n_estimators"]
    base_depth = params["model"]["random_forest"]["max_depth"]
    logger.info(f"Base params from params.yaml -> n_estimators={base_n}, max_depth={base_depth}")

    # Candidate n_estimators values to try
    #    (you can later move this into params.yaml if you want)
    with open("params.yaml") as f:
        params = safe_load(f)
    n_estimators_list = params["hyperparameter_models"]["random_forest"]["n_estimators"]
    depth = params["hyperparameter_models"]["random_forest"]["max_depth"]
    best_mae = float("inf")
    best_mse=  float("inf")
    best_r2= float("-inf")
    best_config = None

    #  Parent run for RF hyperparameter search
    with mlflow.start_run(run_name="rf_hyperparameter_tuning") as parent_run:
        logger.info(f"Parent run id: {parent_run.info.run_id}")
        for n in n_estimators_list:
            for d in depth:
                run_name = f"rf_n_estimators{n}_depth{d}"
                with mlflow.start_run(run_name=run_name, nested=True):
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

                    # Evaluate on validation set
                    logger.info("Evaluating on validation set...")
                    y_pred = model_pipeline.predict(x_val)

                    mae = mean_absolute_error(y_val, y_pred)
                    mse = mean_squared_error(y_val, y_pred)
                    r2 = r2_score(y_val, y_pred)

                    logger.info(f"[{run_name}] MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f}")

                    # Log metrics
                    mlflow.log_metric("MAE", mae)
                    mlflow.log_metric("MSE", mse)
                    mlflow.log_metric("R2_Score", r2)

                    # Log params
                    mlflow.log_param("n_estimators", n)
                    mlflow.log_param("max_depth", d)
                    mlflow.log_param("model_type", "RandomForestRegressor")

                    # Log full sklearn pipeline (if your MLflow+DagsHub combo supports it)
                    mlflow.sklearn.log_model(
                        sk_model=rf_hp,
                        artifact_path="model_pipeline",
                    )

    # Hyperparameter training for xgboost
    with mlflow.start_run(run_name="xgb_hyperparameter_tuning") as parent_run:
        logger.info("starting parent run for xgboost")
        learning_rates = params["hyperparameter_models"]["xgboost"]["learning_rate"]
        n_estimators_list = params["hyperparameter_models"]["xgboost"]["n_estimators"]
        depths = params["hyperparameter_models"]["xgboost"]["max_depth"]
        for lr in learning_rates:
            for n in n_estimators_list:
                for d in depths:
                    run_name=f"xgboost {lr} lr n{n} {d}d(depth)"
                    with mlflow.start_run(run_name=run_name,nested=True):
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
                        logger.info("fitting xgboost model pipeline")
                        model_pipeline.fit(x_train,y_train)
                        logger.info("evaluating on validation set")
                        y_pred=model_pipeline.predict(x_val)
                        mae=mean_absolute_error(y_val,y_pred)
                        mse=mean_squared_error(y_val,y_pred)
                        r2=r2_score(y_val,y_pred)
                        logger.info(f"[{run_name}] MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f}")
                        mlflow.log_metric("MAE",mae)
                        mlflow.log_metric("MSE",mse)
                        mlflow.log_metric("R2_Score",r2)
                        mlflow.log_param("learning_rate",lr)
                        mlflow.log_param("n_estimators",n)
                        mlflow.log_param("max_depth",d)
                        mlflow.log_param("model_type","XGBRegressor")
                        mlflow.sklearn.log_model(
                            sk_model=xgb_hp,
                            artifact_path="model_pipeline",
                        )
