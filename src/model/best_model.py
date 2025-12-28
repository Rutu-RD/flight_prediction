import mlflow
import pandas as pd
import os
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
load_dotenv()
tracking_uri=os.getenv("MLFLOW_TRACKING_URI")

def load_children() -> pd.DataFrame:
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "flight_price_hyperparameter_tuning_experiments" 
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
            model_name="Random_Forest_Flight_Price_Prediction_Model",
            stage="Staging")
    xgb_model_version = register_and_stage_model(
            run_id=best_model_df.iloc[1].run_id,
            model_name="XGBoost_Flight_Price_Prediction_Model",
            stage="Staging")
    print(f"Registered Random Forest model version: {rf_model_version}")
    print(f"Registered XGBoost model version: {xgb_model_version}")


    