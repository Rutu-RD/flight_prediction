import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error 
from yaml import safe_load
logging.basicConfig(level=logging.INFO)
console = logging.StreamHandler()
logger = logging.getLogger(__name__)
logger.addHandler(console)




def build_preprocessor() -> ColumnTransformer:
    categorical_cols = [
        'airline',
        'source_city',
        'destination_city',
        'additional_info'
    ]

    numerical_cols = [
        'total_stops',
        'duration',
        'dep_hr',
        'dep_min',
        'arrival_hr',
        'arrival_min',
        'flight_day',
        'flight_month',
        'day_of_week',
        'is_weekend'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    return preprocessor
if __name__ == "__main__":
    logger.info("encoding features")
    preprocessor = build_preprocessor()
    logger.info("encoding features completed")