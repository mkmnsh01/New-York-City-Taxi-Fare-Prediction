import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

###################################
# Function to add year,month,day,weekday and hour to data
def add_dateparts(df, col):
    df[col + '_year'] = df[col].dt.year
    df[col + '_month'] = df[col].dt.month
    df[col + '_day'] = df[col].dt.day
    df[col + '_weekday'] = df[col].dt.weekday
    df[col + '_hour'] = df[col].dt.hour

# Function to calculate distance between two coordinates
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

# Function to add trip distance
def add_trip_distance(df):
    df['trip_distance'] = haversine_np(df['pickup_longitude'], df['pickup_latitude'], df['dropoff_longitude'], df['dropoff_latitude'])

# Function to add drop distance in data from landmark
def add_landmark_dropoff_distance(df, landmark_name, landmark_lonlat):
    lon, lat = landmark_lonlat
    df[landmark_name + '_drop_distance'] = haversine_np(lon, lat, df['dropoff_longitude'], df['dropoff_latitude'])

# Function to add landmark in data
def add_landmarks(df):
    # Adding coordinates of popular landmark
    jfk_lonlat = -73.7781, 40.6413 # JFK Airport
    lga_lonlat = -73.8740, 40.7769 # LGA Airport
    ewr_lonlat = -74.1745, 40.6895 # EWR Airport
    met_lonlat = -73.9632, 40.7794 # Met Meuseum
    wtc_lonlat = -74.0099, 40.7126 # World Trade Center
    landmarks = [('jfk', jfk_lonlat), ('lga', lga_lonlat), ('ewr', ewr_lonlat), ('met', met_lonlat), ('wtc', wtc_lonlat)]
    for name, lonlat in landmarks:
        add_landmark_dropoff_distance(df, name, lonlat)

# Function to evaluate rmse
def rmse(targets,preds):
    return mean_squared_error(targets,preds,squared=False)

# Function to remove invalid fare amount and other irregular values
def remove_outliers(df):
    return df[(df['fare_amount'] >= 0.) & (df['fare_amount'] <= 500.) &
            (df['pickup_longitude'] >= -75) & (df['pickup_longitude'] <= -72) & 
            (df['dropoff_longitude'] >= -75) & (df['dropoff_longitude'] <= -72) & 
            (df['pickup_latitude'] >= 40) & (df['pickup_latitude'] <= 42) & 
            (df['dropoff_latitude'] >=40) & (df['dropoff_latitude'] <= 42) & 
            (df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)]
###################################

def main():
    # Selecting required columns and assigning data type for them
    selected_cols = 'fare_amount,pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count'.split(',')
    dtypes = {'fare_amount':'float32','pickup_longitude':'float32','pickup_latitude':'float32','dropoff_longitude':'float32','dropoff_latitude':'float32','passenger_count':'uint8'}

    # Load New York Taxi fare data
    df = pd.read_csv('taxi_fare_data.csv',usecols=selected_cols,dtype=dtypes,parse_dates=['pickup_datetime'])

    # Removing irregualar data
    df = remove_outliers(df)
    
    # Spliting test and validation sets
    train_df,val_df = train_test_split(df,test_size=.2,random_state=42)

    # Adding trip distance
    add_trip_distance(train_df)
    add_trip_distance(val_df)

    # Adding parts of date time to data frame
    add_dateparts(train_df,'pickup_datetime')
    add_dateparts(val_df,'pickup_datetime')

    # Adding landmark to data
    add_landmarks(train_df)
    add_landmarks(val_df)

    # Input and target columns for taining model
    input_cols = ['pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', 'passenger_count',
        'pickup_datetime_year', 'pickup_datetime_month', 'pickup_datetime_day','pickup_datetime_weekday', 'pickup_datetime_hour',
            'trip_distance','jfk_drop_distance', 'lga_drop_distance', 'ewr_drop_distance','met_drop_distance', 'wtc_drop_distance']
    target_col = 'fare_amount'

    # Seperating training data inputs and targets from train data frame
    train_inputs = train_df[input_cols]
    train_targets = train_df[target_col]

    # Seperating training data inputs and targets from validate data frame
    val_inputs = val_df[input_cols]
    val_targets = val_df[target_col]

    # Using Ridge model
    ridge_model = Ridge(random_state=42)

    # Traing Ridge model
    ridge_model.fit(train_inputs,train_targets)

    # Predicting from ridge model
    train_preds_ridge_model = ridge_model.predict(train_inputs)
    val_preds_ridge_model = ridge_model.predict(val_inputs)

    # Eval ridge model using rmse
    train_rmse_ridge_model = rmse(train_targets,train_preds_ridge_model)
    val_rmse_ridge_model = rmse(val_targets,val_preds_ridge_model)
    
    print('')
    print('rmse for training data (ridge model): ',train_rmse_ridge_model)
    print('rmse for validation data (ridge model): ',val_rmse_ridge_model,'\n')

    # Using Random forest model
    random_forest_model = RandomForestRegressor(random_state=42,n_estimators=100,max_depth=10,n_jobs=-1)

    # Traing Random forest model
    random_forest_model.fit(train_inputs,train_targets)

    # Predicting from Random forest model
    train_preds_random_forest_model = random_forest_model.predict(train_inputs)
    val_preds_random_forest_model = random_forest_model.predict(val_inputs)

    # Eval Random forest model using rmse
    train_rmse_random_forest_model = rmse(train_targets,train_preds_random_forest_model)
    val_rmse_random_forest_model = rmse(val_targets,val_preds_random_forest_model)

    print('rmse for training data (Random forest model): ',train_rmse_random_forest_model)
    print('rmse for validation data (Random forest model): ',val_rmse_random_forest_model,'\n')

    # Using XGBRegressor model
    xgbreg_model = XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror')

    # Traing XGBRegressor model
    xgbreg_model.fit(train_inputs,train_targets)

    # Predicting from XGBRegressor model
    train_preds_xgbreg_model = xgbreg_model.predict(train_inputs)
    val_preds_xgbreg_model = xgbreg_model.predict(val_inputs)

    # Eval XGBRegressor model using rmse
    train_rmse_xgbreg_model = rmse(train_targets,train_preds_xgbreg_model)
    val_rmse_xgbreg_model = rmse(val_targets,val_preds_xgbreg_model)

    print('rmse for training data (XGBRegressor model): ',train_rmse_xgbreg_model)
    print('rmse for validation data (XGBRegressor model): ',val_rmse_xgbreg_model,'\n')

if __name__ == '__main__':
    main()