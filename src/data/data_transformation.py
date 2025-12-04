import pandas as pd
import numpy as np
import os
import logging

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
       .assign(flight_year=lambda x: x['date_of_journey'].dt.year)
       )
    
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