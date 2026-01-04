import pandas as pd 
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from yaml import safe_load
from src.logger import setup_logger

dataset_logger = setup_logger(name="data_transformation")

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    dataset_logger.info("adding new features is_weekend,day_of_week to dataframe ")
    return(
        df
        #weekeday feature
       .assign(day_of_week=lambda x: x['date_of_journey'].dt.dayofweek)
       .assign(is_weekend=lambda x: x['day_of_week'].isin([5,6]).astype(int))
       .drop(columns=['date_of_journey','Unnamed: 0'] )
       .drop(index=748)
    )





if __name__ == "__main__":
    dataset_logger.info("feature engineering starting")
    try:
        dataset_logger.info("reading data from transformed folder ")  
        df=pd.read_csv(os.path.join( "data","transformed", "transformed_flight_price.csv"), parse_dates=["date_of_journey"])
    except FileNotFoundError as e:
        dataset_logger.error("File not found: data/transformed/transformed_flight_price.csv")
        raise e
   
    engineered_df=feature_engineering(df)
    
    data_path = os.path.join("data", "features")
    os.makedirs(data_path, exist_ok=True)
    engineered_df.to_csv(os.path.join("data","features","feature_engineered_flight_price.csv"), index=False)    
    dataset_logger.info("feature engineering completed")
    
    print(df.head())
    
  