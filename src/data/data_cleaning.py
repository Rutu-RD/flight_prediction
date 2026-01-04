import pandas as pd
import numpy as np
import os
import logging
from src.logger import setup_logger

dataset_logger = setup_logger(name="data_cleaning")


import warnings
warnings.filterwarnings("ignore")

# changing column names
# changing column names
# changing column names
def cleaning(df: pd.DataFrame) -> pd.DataFrame:

    return(
        df
        # changing columns to lower case
        .rename(columns=str.lower)
        # renaming columns
        .rename(columns={'source': 'source_city'})
        .rename(columns={'destination': 'destination_city'})
        .rename(columns={'arrival_time': 'arr_time'})
        .rename(columns={'dept_time': 'dep_time'})
        .assign(**{
            col :df[col].str.strip()
            for col in df.select_dtypes(include='O').columns
        })
        # handling price column
        .assign(price=lambda x: x['price'].astype(float))
        # handling total_stops column
        .assign(total_stops=lambda x: x['total_stops'].replace({'non-stop':0, '1 stop':1, '2 stops':2, '3 stops':3, '4 stops':4}))
        # lowering each row value of airline, source_city, destination_city
        .assign(airline=lambda x: x['airline'].str.lower())
        .assign(source_city=lambda x: x['source_city'].str.lower())
        .assign(destination_city=lambda x: x['destination_city'].str.lower())
        .assign(dep_time=lambda x: pd.to_datetime(x['dep_time'],errors='coerce'))
        
       
        .drop(columns=['Airline','Source','Destination','Route','Date_of_Journey', 'Dep_Time',
       'Arrival_Time', 'Duration', 'Total_Stops', 'Additional_Info'])
     
        .assign(date_of_journey=lambda x :pd.to_datetime(x['date_of_journey']))
        .assign(arr_time=lambda x : x.loc[:,'arr_time'].str.split().str.get(0))
        #.assign(arr_time=lambda x:pd.to_datetime(x['arr_time']).dt.time)
        .assign(arr_time=lambda x:pd.to_datetime(x['arr_time']))
        .assign(duration=lambda x:
                x['duration'].str.extract(r'(\d+)h')[0].fillna(0).astype(int) * 60 +
                x['duration'].str.extract(r'(\d+)m')[0].fillna(0).astype(int)
                )
        .assign(additional_info=lambda x: 
                np.where(x['airline'].isin(['jet airways business', 'vistara business','']),'business',x['additional_info']))
        .assign(additional_info= lambda x: 
                np.where(x['airline'].isin(['multiple carriers premium economy']),'premium',x['additional_info']))
       
        
        )



if __name__ == "__main__":
   
    try:
        df=pd.read_csv(os.path.join( "data","raw", "flight_price.csv"))
    except FileNotFoundError as e:
        dataset_logger.error("File not found: data/raw/flight_price.csv")
        raise e
    dataset_logger.info("data read from raw folder")
    dataset_logger.info("data cleaning starting")
    cleaned_df=cleaning(df)
    #cleaned_df.info()
    dataset_logger.info(" there are total {} rows and {} columns after cleaning".format(cleaned_df.shape[0],cleaned_df.shape[1]))
    
    data_path = os.path.join("data", "interim")
    os.makedirs(data_path, exist_ok=True)
    cleaned_df.to_csv(os.path.join("data","interim","cleaned_flight_price.csv"))    
    dataset_logger.info("data cleaning completed")

