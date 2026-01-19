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
    mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
    mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_username is None or mlflow_password is None or tracking_uri is None:
        logger.error("MLflow tracking credentials are not set in environment variables.")
        return None
    return tracking_uri


def get_mlflow_client():
    return MlflowClient()


def load_children() -> pd.DataFrame:
    tracking_uri = check_credentials()
    if not tracking_uri:
        raise RuntimeError("Missing MLflow credentials. Provide via env vars.")
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = "hyperparameter_tuning RF and xgboost"
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment {experiment_name} not found")

    experiment_id = exp.experiment_id

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        output_format="pandas"
    )

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


def get_best_model(df: pd.DataFrame) -> pd.DataFrame:
    # Higher R2 better, lower MSE better
    xgb_df = df[df['parent_run_name'] == 'xgb_hyperparameter_tuning']
    rf_df = df[df['parent_run_name'] == 'rf_hyperparameter_tuning']

    if rf_df.empty:
        raise ValueError("No RF child runs found in the experiment.")
    if xgb_df.empty:
        raise ValueError("No XGB child runs found in the experiment.")

    rf_df = rf_df.sort_values(by=['metrics.R2_Score', 'metrics.MSE'], ascending=[False, True])
    best_rf = rf_df.iloc[0]

    xgb_df = xgb_df.sort_values(by=['metrics.R2_Score', 'metrics.MSE'], ascending=[False, True])
    best_xgb = xgb_df.iloc[0]

    best_models = pd.DataFrame([best_rf, best_xgb])
    return best_models


def register_model(run_id: str, model_name: str) -> int:
    tracking_uri = check_credentials()
    if not tracking_uri:
        raise RuntimeError("Missing MLflow credentials. Provide via env vars.")
    mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"runs:/{run_id}/model_pipeline"
    registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = int(registered.version)
    return version


def tag_model_version(model_name: str, version: int, tags: dict):
    client = get_mlflow_client()
    for k, v in tags.items():
        client.set_model_version_tag(model_name, str(version), k, str(v))


def get_current_production_metrics(model_name: str):
    """
    Returns dict with current Production model version and its stored metric tags,
    or None if no Production version exists.
    """
    client = get_mlflow_client()
    versions = client.search_model_versions(f"name='{model_name}'")

    prod = None
    for v in versions:
        if getattr(v, "current_stage", None) == "Production":
            prod = v
            break

    if prod is None:
        return None

    
    r2 = float(prod.tags.get("r2", "nan"))
    mse = float(prod.tags.get("mse", "nan"))
    return {"version": int(prod.version), "r2": r2, "mse": mse}


def promote_to_production(model_name: str, version: int):
    client = get_mlflow_client()
    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage="Production",
        archive_existing_versions=True
    )



if __name__ == "__main__":

    tracking_uri = check_credentials()
    if not tracking_uri:
        raise RuntimeError("Missing MLflow credentials. Set MLFLOW_TRACKING_URI/USERNAME/PASSWORD.")
    mlflow.set_tracking_uri(tracking_uri)

    children_merged = load_children()

    # Save runs dataframe for reporting
    data_path = os.path.join("models", "reports", "experiments_run_df")
    os.makedirs(data_path, exist_ok=True)
    children_merged.to_csv(os.path.join(data_path, "hyperparameter_tuning_runs.csv"), index=False)

    best_model_df = get_best_model(children_merged)

    # Print best RF
    logger.info("Best RF child run (highest R2, lowest MSE)")
    logger.info(f"run_id={best_model_df.iloc[0].run_id} mse={best_model_df.iloc[0]['metrics.MSE']} r2={best_model_df.iloc[0]['metrics.R2_Score']}")
    logger.info(f"n_estimators={best_model_df.iloc[0].get('params.n_estimators', None)} max_depth={best_model_df.iloc[0].get('params.max_depth', None)}")

    # Print best XGB
    logger.info("Best XGB child run (highest R2, lowest MSE)")
    logger.info(f"run_id={best_model_df.iloc[1].run_id} mse={best_model_df.iloc[1]['metrics.MSE']} r2={best_model_df.iloc[1]['metrics.R2_Score']}")
    logger.info(f"n_estimators={best_model_df.iloc[1].get('params.n_estimators', None)} max_depth={best_model_df.iloc[1].get('params.max_depth', None)} learning_rate={best_model_df.iloc[1].get('params.learning_rate', None)}")

    # Choose best between RF and XGB (your current rule = higher R2 wins)
    if float(best_model_df.iloc[0]['metrics.R2_Score']) > float(best_model_df.iloc[1]['metrics.R2_Score']):
        best_run_id = best_model_df.iloc[0].run_id
        candidate_algo = "rf"
        candidate_r2 = float(best_model_df.iloc[0]['metrics.R2_Score'])
        candidate_mse = float(best_model_df.iloc[0]['metrics.MSE'])
        logger.info("Random Forest selected as best candidate among RF vs XGB (by R2).")
    else:
        best_run_id = best_model_df.iloc[1].run_id
        candidate_algo = "xgb"
        candidate_r2 = float(best_model_df.iloc[1]['metrics.R2_Score'])
        candidate_mse = float(best_model_df.iloc[1]['metrics.MSE'])
        logger.info("XGBoost selected as best candidate among RF vs XGB (by R2).")

    model_name = "best_model_for_production"

    # 1) Register candidate under ONE stable registry name
    candidate_version = register_model(best_run_id, model_name)
    logger.info(f"Registered candidate under '{model_name}' version={candidate_version} from run_id={best_run_id}")

    # 2) Tag the candidate version with metrics (so future comparisons are easy)
    tag_model_version(model_name, candidate_version, {
        "algo": candidate_algo,
        "r2": candidate_r2,
        "mse": candidate_mse,
        "source_run_id": best_run_id
    })

    # 3) Compare against current Production
    prod = get_current_production_metrics(model_name)
    should_promote = False

    if prod is None:
        logger.info("No existing Production model found. Promoting first candidate to Production.")
        should_promote = True
    else:
        logger.info(f"Current Production: v{prod['version']} r2={prod['r2']} mse={prod['mse']}")
        logger.info(f"Candidate: v{candidate_version} r2={candidate_r2} mse={candidate_mse}")

        # Promotion rule (safe & strict): must improve both
        if candidate_r2 > prod["r2"] and candidate_mse < prod["mse"]:
            should_promote = True
        else:
            logger.info("Candidate is NOT better than Production by strict rule (r2↑ AND mse↓). Not promoting.")

    # 4) Promote only if better
    if should_promote:
        promote_to_production(model_name, candidate_version)
        logger.info(f"Promoted '{model_name}' v{candidate_version} to Production (archived previous Production).")
    else:
        logger.info(f" No promotion performed. '{model_name}' Production remains unchanged.")
