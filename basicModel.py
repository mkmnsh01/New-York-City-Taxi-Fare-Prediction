import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

###################################
# Train Hardcoded (Mean of targets)
class MeanRegressor():
    def fit(self,inputs,targets):
        self.mean = targets.mean()
        return
    def predict(self,inputs):
        return np.full(inputs.shape[0],self.mean)
###################################

###################################
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

    # Load 1% Taxi fare data
    df = pd.read_csv('taxi_fare_data.csv',usecols= selected_cols,dtype=dtypes,parse_dates=['pickup_datetime'])

    # Removing irregular data
    df = remove_outliers(df)

    # Prepare data for trainig
    train_df,val_df = train_test_split(df,test_size=0.2,random_state=42)

    # Removing NA values
    train_df = train_df.dropna()
    val_df = val_df.dropna()

    # Marking input columns and target columns
    input_cols = ['pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', 'passenger_count']
    target_col = 'fare_amount'

    # Seperating training data inputs and targets from train data frame
    train_inputs = train_df[input_cols]
    train_targets = train_df[target_col]

    # Sepreating validate data inputs and targets from validate data frame
    val_inputs = val_df[input_cols]
    val_targets = val_df[target_col]

    # Mean model instance
    mean_model = MeanRegressor()
    mean_model.fit(inputs=train_inputs,targets=train_targets)

    # Predected result for train inputs
    train_preds_mean_model = mean_model.predict(train_inputs)

    # Predected result for validation inputs
    val_preds_mean_model = mean_model.predict(val_inputs)

    # Eval Mean model using rmse
    train_rmse_mean_model = rmse(train_targets,train_preds_mean_model)
    val_rmse_mean_model = rmse(val_targets,val_preds_mean_model)

    print('rmse for training data (mean model): ',train_rmse_mean_model)
    print('rmse for validation data (mean model): ',val_rmse_mean_model)

    # Trying Linear Regression model to predict 
    linearReg_model = LinearRegression()
    linearReg_model.fit(train_inputs,train_targets)

    # Predected result for train inputs using Linear Regression
    train_preds_linear_model = linearReg_model.predict(train_inputs)

    # Predected result for validation inputs using Linear Regression
    val_preds_linear_model = linearReg_model.predict(val_inputs)

    # Eval Linear Model using rmse
    train_rmse_linear_model = rmse(train_targets,train_preds_linear_model)
    val_rmse_linear_model = rmse(val_targets,val_preds_linear_model)

    print('rmse for training data (Linear model): ',train_rmse_linear_model)
    print('rmse for validation data (Linaer model): ',val_rmse_linear_model)


if __name__ == '__main__':
    main()