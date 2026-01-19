from datetime import timedelta
from flask import Flask,render_template,request,jsonify
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.pyfunc
from sklearn.pipeline import Pipeline
from dotenv import load_dotenv
from datetime import datetime
#from src.logger import setup_logger
#logger = setup_logger(name="flask_app")
load_dotenv()



app= Flask(__name__)

def get_credentials():
    mlflow_username=os.getenv("MLFLOW_TRACKING_USERNAME")
    mlflow_password=os.getenv("MLFLOW_TRACKING_PASSWORD")
    tracking_uri=os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_username is None or mlflow_password is None:
        #logger.error("MLflow tracking credentials are not set in environment variables.")
        return False
    return tracking_uri

def feature_engineering(airline,source_city,destination_city,travel_date,dep_time,arrival_time,total_stops,additional_info):
    
    # Convert travel_date to datetime object
    travel_date = datetime.strptime(travel_date, '%Y-%m-%d')
    dep_time = datetime.strptime(dep_time, '%H:%M')
    arrival_time = datetime.strptime(arrival_time, '%H:%M')
    if arrival_time < dep_time:
        arrival_time += timedelta(days=1)

   # Calculate duration
   # Calculate duration (minutes) as int
    duration = float((arrival_time - dep_time).seconds // 60)

    total_stops = float(total_stops)  

    departure_hour = int(dep_time.hour)
    departure_minute = int(dep_time.minute)
    arrival_hr = int(arrival_time.hour)
    arrival_minute = int(arrival_time.minute)

    flight_day = int(travel_date.day)
    flight_month = int(travel_date.month)
    day_of_week = int(travel_date.weekday())
    is_weekend = int(1 if day_of_week >= 5 else 0)
    #logger.info("converting input data to a single row test_dataframe")
    features = {
        'airline': airline,
        'source_city': source_city,
        'destination_city': destination_city,
        'duration': duration,
        'total_stops': total_stops,
        'additional_info': additional_info,
        'arrival_hr': arrival_hr,
        'arrival_min': arrival_minute,
        'flight_day': flight_day,
        'flight_month': flight_month,
        'dep_hr': departure_hour,
        'dep_min': departure_minute,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,

          }
    prediction_df=pd.DataFrame([features])
    #logger.info("Feature engineered DataFrame:")
   # logger.info(prediction_df)
    return prediction_df
    

def get_model(tracking_uri):
    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.pyfunc.load_model(
    "models:/best_model_for_production/3"
    )
    #logger.info("Model loaded successfully from MLflow Model Registry")
    return model

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html') 

@app.route('/predict',methods=['POST'])
def predict():
    
   # logger.info("Model loaded from MLflow Model Registry")

    # getting input from the form
    
    airline=request.form.get('airline')
    source_city=request.form.get('source_city')
    destination_city=request.form.get('destination_city')
    travel_date=request.form.get('journey_date')
    dep_time=request.form.get('departure_time')
    arrival_time=request.form.get('arrival_time')
    total_stops=request.form.get('stops')
    additional_info=request.form.get('additional_info')
    # feature engineering
    # logger.info("starting feature engineering for prediction data")
    features=feature_engineering(airline,source_city,destination_city,travel_date,dep_time,arrival_time,total_stops,additional_info)
    prediction=model.predict(features)
    dict_features=features.to_dict(orient='records')[0]|{'prediction_price': prediction[0]}

    #prediction=prediction['airline','source_city','destination_city','duration','total_stops','additional_info','arrival_hr','arrival_min','flight_day','flight_month','dep_hr','dep_min','day_of_week','is_weekend']
    return jsonify({"prediction": prediction[0]})



if __name__=="__main__":
    tracking_uri=get_credentials()
    model=get_model(tracking_uri)

    app.run(debug=True,host='0.0.0.0',port=5000)