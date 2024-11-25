# This file contains code to evaluate the performance of the model.
# The model is evaluated using the mean squared error (MSE)

import os
import pandas as pd
import numpy as np
import predictor.test_predictor
from predictor.utils import stations_list

# Helper function to get data for a specific station at a specific year.
# Given the year X, we want data from Nov. 1, X-1 to Dec. 31, X.
# Input: station (string), year (int), directory (string)

def get_data_station_year(station, year, directory = "data/restructured_simple/"):
    # Get the data for the specified station and year
    filename = f"{station}.csv"
    file_path = os.path.join(directory, filename)
    df = pd.read_csv(file_path)

    # The first column of the dataframe is the year, second is the month, third is the day
    df = df[((df['YEAR'] == year - 1) & (df['MONTH'] >= 11)) |
            (df['YEAR'] == year)]

    return df

# Helper function to predict t_min, t_avg, t_max for the next five days, given the current day of a specific station.
# Input: station (string), year (int), month (int), day (int), predictor (Predictor object), data (dataframe)
# Output: list of predicted temperatures for the next five days

def predict_station_day(station, year, month, day, predictor, data):
    #predictions = predictor.predict(data)
    # for testing
    predictions = np.zeros(15)
    #print(predictions)
    return predictions

#predict_station_day("KBNA", 2023, 11, 19, predictor.test_predictor.TestPredictor(), None)

# Helper function to get MSE for a specific station and current day for predictions of the next five days.
# Input: station (string), year (int), month (int), day (int), predictor (Predictor object), data (dataframe)
# Output: MSE (float)

def get_mse_station_day(station, year, month, day, predictor, data):
    # Get the data for the specified station and year
    df = get_data_station_year(station, year)
    # get index of current day
    current_day_index = df[(df['YEAR'] == year) & (df['MONTH'] == month) & (df['DAY'] == day)].index[0]
    # grab the following five rows using index
    actual_temps = df.loc[current_day_index:current_day_index+4, ['TMIN', 'TAVG', 'TMAX']].values
    # make into a 1D list
    actual_temps = actual_temps.flatten()
    actual_temps = actual_temps.tolist()
    # Get the predicted temperatures for the next five days
    predicted_temps = predict_station_day(station, year, month, day, predictor, data)
    # Calculate the mean squared error
    #print(actual_temps)
    #print(predicted_temps)
    mse = np.mean((actual_temps - predicted_temps) ** 2)
    #print(mse)
    return mse

# get_mse_station_day("KBNA", 2023, 11, 19, predictor.test_predictor.TestPredictor(), None)


# Evaluate the model on a given station and year
# Input: station (string), year (int), predictor (Predictor object)
# Output: MSE (float)

def evaluate_model_station_year(station, year, predictor):
    # Get the data for the specified station and year
    df = get_data_station_year(station, year)
    # Initialize a list to store the MSE for each day
    mses = []
    # Iterate over each day in November and December of the given year
    for month in range(11, 13):
        if month == 11:
            day_range = range(25, 31)
        else:
            day_range = range(1, 11)
        for day in day_range:
            # Calculate the MSE for the current day
            mse = get_mse_station_day(station, year, month, day, predictor, df)
            #print(mse)
            mses.append(mse)
    #print(mses)
    # Calculate the average MSE for the year
    avg_mse = np.mean(mses)
    return avg_mse

evaluate_model_station_year("KDCA", 2023, predictor.test_predictor.TestPredictor())

# Function to evaluate for all stations in a given year
# Input: year (int), predictor (Predictor object)
# Output: Mean MSE (float)

def evaluate_model_year(year, predictor):
    # Initialize a list to store the MSE for each station
    mse_list = []
    # Iterate over each station
    for station in stations_list:
        # Calculate the MSE for the station and year
        mse = evaluate_model_station_year(station, year, predictor)
        mse_list.append(mse)
    # print(mse_list)
    # Calculate the mean MSE for all stations
    mean_mse = np.mean(mse_list)
    return mean_mse

evaluate_model_year(2017, predictor.test_predictor.TestPredictor())
evaluate_model_year(2018, predictor.test_predictor.TestPredictor())
evaluate_model_year(2019, predictor.test_predictor.TestPredictor())
evaluate_model_year(2020, predictor.test_predictor.TestPredictor())
evaluate_model_year(2021, predictor.test_predictor.TestPredictor())
evaluate_model_year(2022, predictor.test_predictor.TestPredictor())

# Function to evaluate the model for a given amount of years
# Input: start_year (int), end_year (int), predictor (Predictor object)
# Output: List of mean MSE for each year

def evaluate_model_years(start_year, end_year, predictor):
    mse_list = []
    for year in range(start_year, end_year + 1):
        mse = evaluate_model_year(year, predictor)
        mse_list.append(mse)
        print(mse)
    return mse_list

evaluate_model_years(2017, 2022, predictor.test_predictor.TestPredictor())

