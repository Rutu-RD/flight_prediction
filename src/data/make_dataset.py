import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_data(file_path):
    """Reads a CSV file and returns a pandas DataFrame."""
    logging.info(f"Reading data from {file_path}")
    df = pd.read_csv(file_path)
    return df


if __name__ == "__main__":
    # Example usage
    file_path = os.path.join("data", "raw", "flights.csv")
    df = read_data(file_path)
    # Example: Save processed data
    processed_path = os.path.join("data", "processed", "flights_processed.csv")
    df.to_csv(processed_path, index=False)
    logging.info(f"Data has been added to the processed files: {processed_path}")

