import pandas as pd
import numpy as np
import os
import logging
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from xgboost import XGBRegressor
from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import root_mean_squared_error,mean_absolute_error,r2_score

from yaml import safe_load
from src.model.pipeline import build_preprocessor
import joblib
import mlflow
import dagshub
import mlflow.sklearn
from mlflow.models import infer_signature
from dotenv import load_dotenv
load_dotenv()
import warnings
warnings.filterwarnings("ignore")


# setup logging
from src.logger import setup_logger
logger=setup_logger("Baseline_Model_Evaluation")


#check credentials
def check_credentials():
    mlflow_username=os.getenv("MLFLOW_TRACKING_USERNAME")
    mlflow_password=os.getenv("MLFLOW_TRACKING_PASSWORD")
    tracking_uri=os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_username is None or mlflow_password is None:
        logger.error("MLflow tracking credentials are not set in environment variables.")
        return False
    return tracking_uri

#getting data
def get_data():
   logger.info("Getting training and test data")
   X_train = pd.read_csv(os.path.join("data", "splitted_data", "X_train.csv"))
   y_train = pd.read_csv(os.path.join("data", "splitted_data", "y_train.csv"))
   X_test= pd.read_csv(os.path.join("data","splitted_data","X_test.csv"))
   y_test=pd.read_csv(os.path.join("data","splitted_data","y_test.csv"))
   x_val=pd.read_csv(os.path.join("data","splitted_data","X_val.csv"))
   y_val=pd.read_csv(os.path.join("data","splitted_data","y_val.csv"))
   
   if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1 :
        y_train = y_train.iloc[:, 0]
        logger.info("loading train,test,validation data successful")
   else:
        logger.info("error loading data")
       

   return(X_train,y_train,X_test,y_test,x_val,y_val)



def Implement_Model(model_name: str, alphas: float) ->Pipeline:
    logger.info("implementing model:{}".format(model_name))
    if model_name == "LinearRegression":
        model = LinearRegression()
    elif model_name == "RidgeRegression":
        model = Ridge(alpha=alphas)
    elif model_name == "LassoRegression":
        model = Lasso(alpha=alphas)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )
    

    logger.info("made model {}".format(model_name))
    return(model_pipeline)


def model_evaluation(model_name,X_train,y_train,X_test,y_test,X_val,y_val):
    #fitting the model
    logger.info("fitting the model")
    with open("params.yaml") as f:
       params=safe_load(f)
    mlflow.set_experiment("Multiple Baseline Model Evaluations")
    if model_name=="RidgeRegression":
        alpha_list=params['baseline_models']['RidgeRegression']['alpha']
    elif model_name=="LassoRegression":
        alpha_list=params['baseline_models']['LassoRegression']['alpha']
    else:
        alpha_list=params['baseline_models']['LinearRegression']['alpha']

    with mlflow.start_run(run_name=f"model_evaluation_{model_name}") as parent_run:
        for alpha in alpha_list:
            run_name=f"model_evaluation_{model_name}_alpha:{alpha}"
            with mlflow.start_run(run_name=run_name,nested=True) as child_run:
                model_pipeline=Implement_Model(model_name,alpha)
                cv=KFold(n_splits=10, shuffle=True, random_state=42)
                scores=cross_val_score(model_pipeline, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error',error_score="raise")
                cv_train_rmse=-np.mean(scores)
                cv_train_std=np.std(scores)
                mlflow.log_metric("cv_train_rmse", cv_train_rmse)
                mlflow.log_metric("cv_train_std",cv_train_std)
                mlflow.log_param("alpha",alpha)
                mlflow.log_param("model",model_name)

                logger.info("CV RMSE scores for %s: %s", model_name, scores)
                logger.info("cv_train_rmse_%s=%s", model_name, cv_train_rmse)
                logger.info("cv_train_std_%s=%s", model_name, cv_train_std)

                logger.info("Fitting {}".format(model_name))
                model_pipeline.fit(X_train,y_train)
                logger.info("fit completed {}".format(model_name))

                y_pred=model_pipeline.predict(X_val)
                rmse=root_mean_squared_error(y_val,y_pred)
                mae=mean_absolute_error(y_val,y_pred)
                r2=r2_score(y_val,y_pred)
        

                mlflow.log_metric("val_rmse", rmse)
                mlflow.log_metric("val_mae", mae)
                mlflow.log_metric("val_r2", r2)


                signature=infer_signature(X_val.head(10),model_pipeline.predict(X_val.head(10)))
                mlflow.sklearn.log_model(
                sk_model=model_pipeline,
                artifact_path="model_pipeline",
                signature=signature)

                
            

if __name__=="__main__":
   
   try:
      tracking_uri = check_credentials()
      mlflow.set_tracking_uri(tracking_uri)
   except Exception as e:
      logger.error("Failed to set MLflow tracking URI")
      raise e
   X_train,y_train,X_test,y_test,X_val,y_val=get_data()

   logger.info("shape of x_train:{}".format(X_train.shape))
   models_name={"LinearRegression","RidgeRegression","LassoRegression"}
   
    

   for model_name in models_name:
       #model_pipeline,model_name=Implement_Model(model)
      
       model_evaluation(model_name,X_train,y_train,X_test,y_test,X_val,y_val)
       
   

