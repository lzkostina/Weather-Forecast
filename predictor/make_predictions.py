# In this file, we will use a given predictor to make predictions for a given day.


import numpy as np
import os
import datetime
import sys
import pandas as pd
from predictor import test_predictor
from predictor import utils
from analysis.evaluate_model import get_data_station_year
from predictor.utils import stations_list


# Helper function to make predictions for a given day

def make_predictions_station(predictor, year, month, day, station):
    """
    Make predictions for a given day using the specified predictor.

    Args:
        predictor (Predictor): The predictor to use for making predictions.
        year (int): The year of the date for which to make predictions.
        month (int): The month of the date for which to make predictions.
        day (int): The day of the date for which to make predictions.
        station (str): The station for which to make predictions.

    Returns:
        np.array: An array of 15 predicted values.
    """
    # Get the data for the specified station and year
    data = get_data_station_year(station, year, "data/restructured_simple/combined/")
    # Find current day index
    current_day_index = data[(data['YEAR'] == year) & (data['MONTH'] == month) & (data['DAY'] == day)].index[0]
    # Get the data up to the current day, not including the current day
    data = data.loc[:current_day_index - 1]
    # print(data)
    # Get the predicted temperatures for the next five days
    predicted_temps = predictor.predict(data, station)
    return predicted_temps

# make_predictions_station(test_predictor.PreviousDayPredictor(), 2023, 11, 19, "KBNA")

# Make predictions for a given day, for all stations

def make_predictions_all_stations(predictor, year, month, day):
    """
    Make predictions for a given day using the specified predictor for all stations.

    Args:
        predictor (Predictor): The predictor to use for making predictions.
        year (int): The year of the date for which to make predictions.
        month (int): The month of the date for which to make predictions.
        day (int): The day of the date for which to make predictions.

    Returns:
        np.array: An array of 300 predicted values.
    """
    all_predictions = None
    for station in stations_list:
        predictions = make_predictions_station(predictor, year, month, day, station)
        print(station)
        print(predictions)
        if all_predictions is None:
            all_predictions = predictions
        else:
            all_predictions = np.concatenate((all_predictions, predictions))

    return all_predictions

# make_predictions_all_stations(test_predictor.PreviousDayPredictor(), 2023, 11, 19)