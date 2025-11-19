import pandas as pd
import numpy as np
import os

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_data(file_path):
    """Reads a CSV file and returns a pandas DataFrame."""
    logging.info(f"Reading data from {file_path}")
    df = pd.read_csv(file_path)
    return df


if __name__ == "__main__":
    # Example usage
    file_path = os.path.join("data", "external", "flight_price.csv")
    print(file_path)
    df = read_data(file_path)
    # Example: Save processed data
    
    data_path = os.path.join("data", "raw")
    os.makedirs(data_path, exist_ok=True)

    df.to_csv(os.path.join(data_path, "flight_price.csv"), index=False)
    logging.info(f"Data has been added from external folder to raw files: {data_path}")
