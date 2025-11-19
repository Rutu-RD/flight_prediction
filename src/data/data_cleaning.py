import pandas as pd
import numpy as np
import os
import logging
logging.basicConfig(level=logging.INFO)
console = logging.StreamHandler()
logger = logging.getLogger(__name__)
logger.addHandler(console)




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
        # extracting dept_hour and dept_min from dept_time
        .assign(dep_hr=lambda x: x['dep_time'].dt.hour)
        .assign(dep_min=lambda x: x['dep_time'].dt.minute)
        .drop(columns=['Airline','Source','Destination','Route','Date_of_Journey', 'Dep_Time',
       'Arrival_Time', 'Duration', 'Total_Stops', 'Additional_Info'])
        #changing values in airline column as it has jet airline business and jet airway so changing jet airway business to jet airway
        .assign(airline =lambda x:x['airline'].replace({'jet airways business':'jet airways',
                                                      'vistara premium economy':'vistara',
                                                      'multiple carriers premium economy':'multiple carriers'}))
        .assign(date_of_journey=lambda x :pd.to_datetime(x['date_of_journey']))
        .assign(arr_time=lambda x : x.loc[:,'arr_time'].str.split().str.get(0))
        #.assign(arr_time=lambda x:pd.to_datetime(x['arr_time']).dt.time)
        .assign(arr_time=lambda x:pd.to_datetime(x['arr_time']))
        .assign(duration=lambda x:
                x['duration'].str.extract(r'(\d+)h')[0].fillna(0).astype(int) * 60 +
                x['duration'].str.extract(r'(\d+)m')[0].fillna(0).astype(int) 
                )

        
        
    )


if __name__ == "__main__":
    logger.info("data cleaning starting")
    df=pd.read_csv(os.path.join( "data","raw", "flight_price.csv"))
    cleaned_df=cleaning(df)
    data_path = os.path.join("data", "interim")
    os.makedirs(data_path, exist_ok=True)
    cleaned_df.to_csv(os.path.join("data","interim","cleaned_flight_price.csv"), index=False)    
    logger.info("data cleaning completed")
