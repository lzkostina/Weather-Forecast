import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
import joblib
from predictor.utils import stations_list

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

class LinearRegressionPredictor(Predictor):

    def __init__(self, model_path="predictor/models/LinearRegression/"):
        self.model = None
        self.model_path = model_path

    def train_and_save_model(self, model_path):
        """
        Train a linear regression model and save it.
        """
        # Train the model
        for station in stations_list:
            # print station
            print(station)

            # Get the data for the specified station
            station_data = pd.read_csv(f"analysis/regression_data/{station}.csv")
            # The first 153 columns are the features, the last 15 columns are the labels
            X = station_data.iloc[:, 3:153]
            y = station_data.iloc[:, 153:]


            # Train the model
            model = LinearRegression()
            model.fit(X, y)

            # Save the model
            joblib.dump(model, os.path.join(model_path, f"{station}.joblib"))

    def load_model(self, model_path, station):
        """
        Load a linear regression model.
        """
        self.model = joblib.load(os.path.join(model_path, f"{station}.joblib"))

    def transform_data_to_predict(self, data):
        # Get the previous 30 days data of TMIN, TAVG, TMAX, SNOW, PRCP, flattened into a 1D array
        previous_data = data[['TMIN', 'TAVG', 'TMAX', 'SNOW', 'PRCP']].tail(30).values
        # print(np.shape(previous_data))
        previous_data = previous_data.flatten()
        return previous_data

    def predict(self, data, station):
        # Load model for station
        self.load_model(self.model_path, station)
        # Transform data
        X = self.transform_data_to_predict(data)
        # X = np.zeros(150)
        X = X.reshape(1, -1)
        # Predict
        predictions = self.model.predict(X)
        return predictions.reshape(-1)

# Train Linear Regression model
model = LinearRegressionPredictor()
#model.train_and_save_model("predictor/models/LinearRegression/")
# model.predict("data/restructured_simple/KBNA.csv", "KBNA")

# summarize the model
# print(model.summary)


# Example usage
#predictor = PreviousDayPredictor("data/processed/openweather_hourly")
# predictor._load_data()
#print(predictor.predict("2023-11-19"))

