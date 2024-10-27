from decimal import Decimal
from flask import Flask, jsonify
import pickle
import psycopg2
from configparser import ConfigParser
import numpy as np
import pandas as pd
import gzip
import joblib  # used joblib when droppping.
from collections import deque


app = Flask(__name__)
# Initialize the veriables used
last_id = 0
direction = 0
Cloudcover=0
Temp = 0
UV=0
rainfall=0
time = 0
min_temp = 0  
max_temp = 0  
temp_list = [] 
time_list=[]
rainfall_list=[]
 #####################Loading Data #############################
# Load configuration from file


# Load KNN model from file. This will be the most accurate model
try:
    with open('Second Step/knn_model.pkl', 'rb') as f:
        knn_model = pickle.load(f)
    if not hasattr(knn_model, 'predict'):
        raise TypeError("Model did not load correctly. See type")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    knn_model = None

############### Config loading ###################
##################################################
with open('Second Step/config.txt', 'r') as file:
    app.config['SECRET_KEY'] = file.read().strip()  # No hardcoded secret key on Github

def load_config(filename='Second Step/database.ini', section='postgresql'):
    #Load database configuration from a .ini file. Much better security
    parser = ConfigParser()
    parser.read(filename)
    config = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            config[param[0]] = param[1]
    else:
        raise Exception(f"Section {section} not found in the {filename} file")
    return config

#################Connection to DB #################
def connect_to_db(config):
    #Establish connection to the PostgreSQL database.
    try:
        conn = psycopg2.connect(**config)
        print("Connected to the PostgreSQL server.")
        return conn
    except psycopg2.DatabaseError as error:
        print(f"Database connection error: {error}")
        return None


############fetching line by line on DB ############
def fetch_one_row(conn, last_id):
    #Fetch the next row from the database based on the last processed id_num. increment through DB as data is stored.
    with conn.cursor() as cursor:
        #print(last_id)
        cursor.execute("SELECT * FROM weather_data WHERE id_num = %s;", (last_id,))
        row = cursor.fetchone()  # Fetches only the row where id_num equals last_id

        if row:
            # Update last processed ID (assuming ID is in the first column)
            last_id = row[22] 
            last_id = last_id+1
            direction = row[4]
            windspeed=row[3]
            Cloudcover=row[17]
            Temp = row[1]
            UV=row[21]
            time = row[0]

            column_names = [desc[0] for desc in cursor.description] 
            #print(column_names)
            #print(row)
            row_data = dict(zip(column_names, row))
            # Create a mapping from database column names to desired feature names
            feature_mapping = {
                'basel_wind_speed_100m': 'Basel Wind Speed [100 m]',
                'basel_wind_direction_100m': 'Basel Wind Direction [100 m]',
                'basel_wind_speed_900mb': 'Basel Wind Speed [900 mb]',
                'basel_wind_direction_900mb': 'Basel Wind Direction [900 mb]',
                'basel_wind_speed_850mb': 'Basel Wind Speed [850 mb]',
                'basel_wind_direction_850mb': 'Basel Wind Direction [850 mb]',
                'basel_wind_speed_800mb': 'Basel Wind Speed [800 mb]',
                'basel_wind_direction_800mb': 'Basel Wind Direction [800 mb]',
                'basel_wind_speed_700mb': 'Basel Wind Speed [700 mb]',
                'basel_wind_speed_500mb': 'Basel Wind Speed [500 mb]',
                'basel_wind_speed_250mb': 'Basel Wind Speed [250 mb]',
                'basel_cloud_cover_total': 'Basel Cloud Cover Total',
                'basel_cloud_cover_high': 'Basel Cloud Cover High [high cld lay]',
                'basel_cloud_cover_medium': 'Basel Cloud Cover Medium [mid cld lay]',
                'basel_cloud_cover_low': 'Basel Cloud Cover Low [low cld lay]',
                'basel_longwave_radiation': 'Basel Longwave Radiation'
            }
            filtered_data = {feature_mapping[key]: float(value) for key, value in row_data.items() if key in feature_mapping}
          
            # Print or process the filtered data
            #print(filtered_data)
            return filtered_data,last_id,Cloudcover,Temp,direction,UV,time,windspeed
        else:
            print("No new data to process.")
            return None

def load_scaler():
    #Load a scaler object from a gzipped file.
    with gzip.open('Second Step/scaler.gz', 'rb') as f:
        scaler = joblib.load(f)  # use pickle.load(f) if you saved it with pickle
    return scaler

def predict(data_row):
    #Run prediction on a single row of data.
    if not data_row:
        return {"error": "No data available for prediction"}
    normalization_order = [
        'Basel Relative Humidity [2 m]', 'Basel Wind Gust', 'Basel Wind Speed [10 m]', 'Basel Wind Direction [10 m]',
        'Basel Wind Speed [100 m]', 'Basel Wind Direction [100 m]', 'Basel Wind Speed [900 mb]',
        'Basel Wind Direction [900 mb]', 'Basel Wind Speed [850 mb]', 'Basel Wind Direction [850 mb]',
        'Basel Wind Speed [800 mb]', 'Basel Wind Direction [800 mb]', 'Basel Wind Speed [700 mb]',
        'Basel Wind Speed [500 mb]', 'Basel Wind Speed [250 mb]', 'Basel Cloud Cover Total',
        'Basel Cloud Cover High [high cld lay]', 'Basel Cloud Cover Medium [mid cld lay]',
        'Basel Cloud Cover Low [low cld lay]', 'Basel Longwave Radiation'
    ]
    default_values = { #Normilisation occured before I tested for most accurate model. Right now i am setting default values  in order to normilise and then remove it once complete to train 
        'Basel Wind Speed [100 m]': 0.0, 'Basel Wind Direction [100 m]': 0.0,
        'Basel Wind Speed [900 mb]': 0.0, 'Basel Wind Direction [900 mb]': 0.0,
        'Basel Wind Speed [850 mb]': 0.0, 'Basel Wind Direction [850 mb]': 0.0,
        'Basel Wind Speed [800 mb]': 0.0, 'Basel Wind Direction [800 mb]': 0.0,
        'Basel Wind Speed [700 mb]': 0.0, 'Basel Wind Speed [500 mb]': 0.0,
        'Basel Wind Speed [250 mb]': 0.0, 'Basel Cloud Cover Total': 0.0,
        'Basel Cloud Cover High [high cld lay]': 0.0, 'Basel Cloud Cover Medium [mid cld lay]': 0.0,
        'Basel Cloud Cover Low [low cld lay]': 0.0, 'Basel Longwave Radiation': 0.0,
        'Basel Relative Humidity [2 m]': 0.0, 'Basel Wind Direction [10 m]': 0.0,
        'Basel Wind Gust': 0.0, 'Basel Wind Speed [10 m]': 0.0
    }
    final_order = [
        'Basel Wind Speed [100 m]', 'Basel Wind Direction [100 m]','Basel Wind Speed [900 mb]',
        'Basel Wind Direction [900 mb]','Basel Wind Speed [850 mb]','Basel Wind Direction [850 mb]',
        'Basel Wind Speed [800 mb]','Basel Wind Direction [800 mb]', 'Basel Wind Speed [700 mb]',
        'Basel Wind Speed [500 mb]','Basel Wind Speed [250 mb]','Basel Cloud Cover Total',
        'Basel Cloud Cover High [high cld lay]','Basel Cloud Cover Medium [mid cld lay]','Basel Cloud Cover Low [low cld lay]',
        'Basel Longwave Radiation'
    ]
    # Set default values for missing features
 #   print('data row')
 #   print(data_row)
    for feature in normalization_order:
        if feature not in data_row:
            data_row[feature] = default_values[feature]

    # Create DataFrame in normalization order
    features_df = pd.DataFrame([data_row], columns=normalization_order)

    # Normalize features
    scaler = load_scaler()  # Ensure this function is defined
    normalized_features = scaler.transform(features_df)
    normalized_df = pd.DataFrame(normalized_features, columns=normalization_order)
    # Reorder for prediction
    reverted_features_df = normalized_df[final_order]
    # Prediction
    prediction = knn_model.predict(reverted_features_df)  
    return float(prediction[0])

####################panel beat########################
####################################################
def temperature_range(current_temp, temp_list):
    try:
        current_temp =float(current_temp) 
    except:
        raise ValueError("Tempreture is not value to convert to float") 
    if len(temp_list) <= 0:
        temp_list.append(current_temp)  # Initialize with current temperature
    elif len(temp_list) >= 12:
        # If the array has 12 values, remove the oldest (first) value
            temp_list.pop(0)  # Removing the first element (oldest temperature)
            temp_list.append(current_temp)
    else:
        temp_list.append(current_temp)
    # Find the lowest and highest temperature in the array
    print(temp_list)
    min_temp = min(temp_list)
    max_temp = max(temp_list)
    return min_temp, max_temp,temp_list

def rainfallfunc(rainfall,rainfall_list): #used in the graph to display values correctly print(rainfall_list)print(rainfall)
    if len(rainfall_list) <= 0:
        #print(len(rainfall_list))
        rainfall_list.append(rainfall)  # Initialize with current temperature
    elif len(rainfall_list) >= 6:
        # If the array has 6 values, remove the oldest (first) value
            rainfall_list.pop(0)  # Removing the first element (old predictions)
            rainfall_list.append(rainfall)
    else:
        rainfall_list.append(rainfall)
    #print("rainfall list",rainfall_list)
    return rainfall_list

def timegraph(datetime_val, time_list):
    # Attempt to format the datetime value to a string
    try:
        time_str = datetime_val.strftime('%H:%M:%S')
    except AttributeError:
        raise ValueError("Invalid datetime_val. Must be a datetime object.")

    print(time_str)  # Debugging output for the formatted time string

    # Manage time_list based on its length
    if len(time_list) == 0:
        time_list.append(time_str)  # Initialize with the current time
    elif len(time_list) >= 6:
        # If the list has 6 values, remove the oldest (first) value
        time_list.pop(0)  # Remove the oldest time entry
        time_list.append(time_str)  # Add the new time
    else:
        time_list.append(time_str)  # Add the new time if the list has less than 6 values

    # Return the updated time_list
    return time_list

##############################################
################# Wind DirectioN ##############
def get_direction(degree):
    # Error catching
    if not (0 <= degree <= 360):
        raise ValueError("Degree must be between 0 and 360.")
    #print(degree)
    # Getting Directions
    if degree == 0 or degree == 360:
        return "North"
    elif degree == 90:
        return "East"
    elif degree == 180:
        return "South"
    elif degree == 270:
        return "West"
    elif 0 < degree < 90:
        return "North-East"
    elif 90 < degree < 180:
        return "South-East"
    elif 180 < degree < 270:
        return "South-West"
    elif 270 < degree < 360:
        return "North-West"

@app.route('/api/update', methods=['GET'])
def update_data():
    #API endpoint to get a prediction from the next row in the database.
    global last_id
    global temp_list
    global time_list
    global rainfall_list
    #DB connect
    conn = connect_to_db(load_config())
    if conn is None:
        return jsonify({"error": "Failed to connect to the database"}), 500

    row,last_id,Cloudcover,Temp,direction,UV,time,windspeed = fetch_one_row(conn, last_id)
    text = get_direction(direction)
    min_temp, max_temp,temp_list = temperature_range(Temp,temp_list)
    if row is None:
        return jsonify({"error": "No new data to predict"}), 404
    prediction = predict(row)
    rainfall_list = rainfallfunc(prediction,rainfall_list)
    time_list = timegraph(time,time_list)
    conn.close()
    return jsonify({
        "prediction": rainfall_list,
        "direction": text,
        "windspeed":windspeed,
        "cloudCover": Cloudcover,
        "temperature": Temp,
        "uvRadiation": UV,
        "time": time_list,
        "min_temp":min_temp,
        "max_temp":max_temp
    }), 200

if __name__ == '__main__':
    config = load_config()
    conn = connect_to_db(config)
    if conn:
        fetch_one_row(conn, last_id)
        conn.close()
    app.run(debug=True)
    