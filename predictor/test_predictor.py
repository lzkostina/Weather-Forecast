import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from abc import ABC, abstractmethod

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

class Predictor(ABC): 
    @abstractmethod
    def predict(self, data, station):
        """
        Abstract method to be implemented by subclasses to perform predictions.

        Predict function should return a list of 300 numbers in an np array.
        """
        pass

class TestPredictor(Predictor):
    def __init__(self):
        pass

    def predict(self, data, station):

        
        # Generate 15 numbers of the form "XX.X" (all zeroes with one decimal place)
        predictions = np.zeros(15)
        #predictions_rounded = np.around(predictions, 1)
        
        # Create the output string
        #output = f'"{current_date}" is the current date and then {", ".join(predictions_rounded)}'
        
        return predictions


class PreviousDayPredictor(Predictor):
    """
    A predictor that uses the previous day's temperature data to predict the temperature for the next 5 days.
    """

    def __init__(self):
        pass

    def predict(self, data, station):
        # take the last row of data, and get TMIN, TAVG, TMAX
        last_row = data.iloc[-1]
        predictions = np.array([last_row["TMIN"], last_row["TAVG"], last_row["TMAX"]])

        # Repeat the predictions for the next 5 days
        predictions = np.tile(predictions, 5)
        return predictions

class AverageLastWeekPredictor(Predictor):
    """
    """

    def __init__(self):
        pass

    def predict(self, data, station):
        # take the last week of data
        last_week = data.iloc[-7:]
        # get the average of TMIN, TAVG, TMAX
        predictions = last_week[["TMIN", "TAVG", "TMAX"]].mean().values
        # Repeat the predictions for the next 5 days
        predictions = np.tile(predictions, 5)
        return predictions

# Example usage
#predictor = PreviousDayPredictor("data/processed/openweather_hourly")
# predictor._load_data()
#print(predictor.predict("2023-11-19"))

