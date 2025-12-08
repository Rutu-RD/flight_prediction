import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from yaml import safe_load
from sklearn.pipeline import Pipeline
import joblib
logging.basicConfig(level=logging.INFO)
console = logging.StreamHandler()
logger = logging.getLogger(__name__)
logger.addHandler(console)
if __name__ == "__main__":
    logger.info("model evaluation starting")

    x_test = pd.read_csv(os.path.join("data", "splitted_data", "X_val.csv"))
    y_test = pd.read_csv(os.path.join("data", "splitted_data", "y_val.csv"))

    if isinstance(y_test, pd.DataFrame) and y_test.shape[1] == 1:
        y_test = y_test.iloc[:, 0]

    model_pipeline: Pipeline = joblib.load(os.path.join("models", "random_forest_model.pkl"))

    y_pred = model_pipeline.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Mean Absolute Error: {mae}")
    logger.info(f"Mean Squared Error: {mse}")
    logger.info(f"R^2 Score: {r2}")


    logger.info("model evaluation completed")