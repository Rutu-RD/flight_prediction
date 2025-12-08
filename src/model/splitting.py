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

def splitting_data(df:pd.DataFrame)-> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("splitting data into X and y")
    X = df.drop(columns=['price'])
    y = df['price']
    return X, y   


if __name__ == "__main__":
    logger.info("data splitting starting")
    df=pd.read_csv(os.path.join( "data","features", "feature_engineered_flight_price.csv"))
    X, y = splitting_data(df)
    with open("params.yaml") as f:
        params = safe_load(f)
    test_size = params['test_size']
    x_, x_test, y_, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    logger.info("data splitting completed")
    
    x_train, x_val, y_train, y_val = train_test_split(x_, y_, test_size=test_size, random_state=42)
    os.makedirs(os.path.join("data","splitted_data"), exist_ok=True)
    x_train.to_csv(os.path.join("data","splitted_data","X_train.csv"), index=False)
    x_val.to_csv(os.path.join("data","splitted_data","X_val.csv"), index=False)
    y_train.to_csv(os.path.join("data","splitted_data","y_train.csv"), index=False)
    y_val.to_csv(os.path.join("data","splitted_data","y_val.csv"), index=False)
    x_test.to_csv(os.path.join("data","splitted_data","X_test.csv"), index=False)
    y_test.to_csv(os.path.join("data","splitted_data","y_test.csv"), index=False)