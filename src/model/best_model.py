import logging
import mlflow
import pandas as pd
import os
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
load_dotenv()
import warnings
warnings.filterwarnings("ignore")

from src.logger import setup_logger

logger = setup_logger(name="best_model")
def check_credentials():
    mlflow_username=os.getenv("MLFLOW_TRACKING_USERNAME")
    mlflow_password=os.getenv("MLFLOW_TRACKING_PASSWORD")
    tracking_uri=os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_username is None or mlflow_password is None:
        logger.error("MLflow tracking credentials are not set in environment variables.")
        return False
    return tracking_uri



def load_children() -> pd.DataFrame:
    tracking_uri = check_credentials()
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "hyperparameter_tuning RF and xgboost"
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment {experiment_name} not found")
    experiment_id = exp.experiment_id
    #gettting all runs for the experiment
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        output_format="pandas"
    )
    #filtering only child runs
    parents_df = runs_df[runs_df['tags.mlflow.parentRunId'].isna()]
    children_df = runs_df[runs_df['tags.mlflow.parentRunId'].notna()]
    parents_short = parents_df[['run_id', 'tags.mlflow.runName']].rename(
    columns={
        'run_id': 'parent_run_id',
        'tags.mlflow.runName': 'parent_run_name'
    }
    )

    children_merged = children_df.merge(
    parents_short,
    left_on='tags.mlflow.parentRunId',
    right_on='parent_run_id',
    how='left'
    )
    return children_merged

def get_best_model(df) -> pd.DataFrame:
    # Assuming lower MSE and higher R2 are better
    xgb_df=df[df['parent_run_name']=='xgb_hyperparameter_tuning']
    rf_df=df[df['parent_run_name']=='rf_hyperparameter_tuning']
    rf_df=rf_df.sort_values(by=['metrics.R2_Score','metrics.MSE'], ascending=[False,True])
    best_rf=rf_df.iloc[0]
    xgb_df=xgb_df.sort_values(by=['metrics.R2_Score','metrics.MSE'], ascending=[False,True])
    best_xgb=xgb_df.iloc[0]
    best_models=pd.DataFrame([best_rf,best_xgb])
    return best_models

def register_and_stage_model(run_id: str, model_name: str, stage: str = "Staging"):
    tracking_uri = check_credentials()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    
    model_uri = f"runs:/{run_id}/model_pipeline"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    version = registered_model.version

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage
    )

    print(f"[{model_name}] run_id={run_id} version={version} → {stage}")
    return version

if __name__ == "__main__":

    tracking_uri = check_credentials()
    children_merged = load_children()
    
    
    data_path=os.path.join("models","reports","experiments_run_df")
    os.makedirs(data_path, exist_ok=True)
    children_merged.to_csv(os.path.join(data_path, "hyperparameter_tuning_runs.csv"), index=False)

    best_model_df=get_best_model(children_merged)


    print("best model based on lowest mse and highest r2  for random forest" )
    print("run_id", best_model_df.iloc[0].run_id)
    print("mse", best_model_df.iloc[0]['metrics.MSE'])
    print("r2", best_model_df.iloc[0]['metrics.R2_Score'])
    print("n_estimators", best_model_df.iloc[0]['params.n_estimators'] )
    print("max_depth", best_model_df.iloc[0]['params.max_depth'] )

    print("-------------------------------")
    print("best model based on lowest mse and highest r2 for xgboost" )
    print("run_id", best_model_df.iloc[1].run_id)
    print("mse", best_model_df.iloc[1]['metrics.MSE'])
    print("r2", best_model_df.iloc[1]['metrics.R2_Score'])
    print("n_estimators", best_model_df.iloc[1]['params.n_estimators'] )
    print("max_depth", best_model_df.iloc[1]['params.max_depth'] )
    print("learning_rate", best_model_df.iloc[1]['params.learning_rate'] )

    rf_model_version = register_and_stage_model(
            run_id=best_model_df.iloc[0].run_id,
            model_name="rf model :" + best_model_df['tags.mlflow.runName'].iloc[0],
            stage="Staging")
    logger.info(f"Registered Random Forest model version: {rf_model_version}")
    xgb_model_version = register_and_stage_model(
            run_id=best_model_df.iloc[1].run_id,
            model_name="xgb model :" + best_model_df['tags.mlflow.runName'].iloc[1],
            stage="Staging")
    
  
    logger.info(f"Registered XGBoost model version: {xgb_model_version}")
    logger.info("comparing best models of the registered models to use for production")
    if best_model_df.iloc[0]['metrics.R2_Score'] > best_model_df.iloc[1]['metrics.R2_Score']:
        best_run_id=best_model_df.iloc[0].run_id
        logger.info("Random Forest model selected as the best model for production")
        best_model_name = "rf model :" + best_model_df['tags.mlflow.runName'].iloc[0]
        best_model_version = rf_model_version
    else:
        logger.info("XGBoost model selected as the best model for production")
        best_run_id=best_model_df.iloc[1].run_id
        best_model_name = "xgb model :" + best_model_df['tags.mlflow.runName'].iloc[1]
        best_model_version = xgb_model_version
    logger.info(f"Best model for production: {best_model_name}, version: {best_model_version}")
   
    best_model=register_and_stage_model(
        run_id=best_run_id,
        model_name="best_model_for_production",
        stage="Production"
    )
    logger.info(f"Registered Best model for production version: {best_model}")
   


    