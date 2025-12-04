import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from yaml import safe_load

logging.basicConfig(level=logging.INFO)
console = logging.StreamHandler()
logger = logging.getLogger(__name__)
logger.addHandler(console)

def transformation(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("adding arrival_hr, arrival_min, flight_day, flight_month, flight_year columns to dataframe")
    return(
    
    df .assign(arrival_hr=lambda x: x['arr_time'].dt.hour)
       .assign(arrival_min=lambda x: x['arr_time'].dt.minute)
       .assign(flight_day=lambda x: x['date_of_journey'].dt.day)
       .assign(flight_month=lambda x: x['date_of_journey'].dt.month)
       
       )

def spliiting_data(df:pd.Dataframe)-> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("splitting data into X and y")
    X = df.drop(columns=['price'])
    y = df['price']
    return X, y   
if __name__ == "__main__":
    logger.info("data transformation starting")
    
    df=pd.read_csv(os.path.join( "data","interim", "cleaned_flight_price.csv"), parse_dates=["date_of_journey", "dep_time", "arr_time"])
    
    df.info()
    transformed_df=transformation(df)
    
    print("transformed_df_info")
    #transformed_df.info()
    data_path = os.path.join("data", "transformed")
    os.makedirs(data_path, exist_ok=True)
    transformed_df.to_csv(os.path.join("data","transformed","transformed_flight_price.csv"), index=False)    
    logger.info("data transformation completed")
    logger.info("data splitting starting")
    X, y = spliiting_data(transformed_df)
    with open("params.yaml") as f:
        params = safe_load(f)
    test_size = params['test_size']
    x_, x_test, y_, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    logger.info("data splitting completed")
    
    x_train, x_val, y_train, y_val = train_test_split(x_, y_, test_size=test_size, random_state=42)
    os.makedirs(os.path.join("data","split"), exist_ok=True)
    x_train.to_csv(os.path.join("data","split","X_train.csv"), index=False)
    x_val.to_csv(os.path.join("data","split","X_val.csv"), index=False)
    y_train.to_csv(os.path.join("data","split","y_train.csv"), index=False)
    y_val.to_csv(os.path.join("data","split","y_val.csv"), index=False)