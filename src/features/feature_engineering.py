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

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("adding new features to training dataframe")
    return(
        df
        #weekeday feature
       .assign(day_of_week=lambda x: x['date_of_journey'].dt.dayofweek)
       .assign(is_weekend=lambda x: x['day_of_week'].isin([5,6]).astype(int))
       .drop(columns=['date_of_journey','Unnamed: 0'] )
       .drop(index=748)
    )





if __name__ == "__main__":
    logger.info("feature engineering starting")
    df=pd.read_csv(os.path.join( "data","transformed", "transformed_flight_price.csv"), parse_dates=["date_of_journey"])
    engineered_df=feature_engineering(df)
    
    data_path = os.path.join("data", "features")
    os.makedirs(data_path, exist_ok=True)
    engineered_df.to_csv(os.path.join("data","features","feature_engineered_flight_price.csv"), index=False)    
    logger.info("feature engineering completed")
    
    print(df.head())
    
  