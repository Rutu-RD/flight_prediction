import pandas as pd
import numpy as np
import os
import logging
from src.logger import setup_logger

dataset_logger = setup_logger(name="make_dataset")



def read_data(file_path):
    """Reads a CSV file and returns a pandas DataFrame."""
    
    df = pd.read_csv(file_path)
    rows,columns=df.shape
    dataset_logger.info(f'{file_path} data read having {df.shape[0]} rows and {df.shape[1]} columns')
    return df


if __name__ == "__main__":
    # Example usage
    file_path = os.path.join("data", "external", "flight_price.csv")
    print(file_path)
    try:
        dataset_logger.info(f"Reading data from {file_path}")
        df = read_data(file_path)
       
    except FileNotFoundError as e:
        dataset_logger.error(f"File not found: {file_path}")
        raise e
   
    # Example: Save processed data
    
    data_path = os.path.join("data", "raw")
    os.makedirs(data_path, exist_ok=True)

    df.to_csv(os.path.join(data_path, "flight_price.csv"), index=False)
    dataset_logger.info(f"Data has been added from external folder to raw files: {data_path}")
