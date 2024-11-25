import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression, RidgeCV, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import joblib
from predictor.utils import stations_list

import warnings
warnings.filterwarnings("ignore")


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

class RidgeRegressionPredictor(Predictor):

    def __init__(self, model_path="predictor/models/RidgeRegression/"):
        self.model = None
        self.model_path = model_path

    def train_and_save_model(self, model_path):
        """
        Train a linear regression model and save it.
        """

        alphas = [0.1, 1.0, 10.0, 100.0]

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
            model = RidgeCV(alphas=alphas, store_cv_values=True)
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

class LassoPredictor(Predictor):
    def __init__(self, model_path="predictor/models/LassoCV/"):
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        self.model = None
        self.model_path = os.path.join(self.repo_root, model_path)

    def train_and_save_model(self):
        """
        Train a Lasso Regression model with cross-validation for each station and save it.
        """
        # Ensure the model path exists
        os.makedirs(self.model_path, exist_ok=True)

        for station in stations_list:
            print(f"Training Lasso model for station: {station}")
            station_file_path = os.path.join(self.repo_root, f"analysis/regression_data/{station}.csv")

            # Load the data for the specified station
            station_data = pd.read_csv(station_file_path)
            # The first 153 columns are the features, the last 15 columns are the labels
            X = station_data.iloc[:, 3:153]
            y = station_data.iloc[:, 153:]

            # Train the LassoCV model with default parameters
            # Train the Lasso model
            model = Lasso(alpha=0.05, random_state=42)
            model.fit(X, y)

            # Save the trained model
            joblib.dump(model, os.path.join(self.model_path, f"{station}.joblib"))
            print(f"Model for station {station} saved successfully!")

    def load_model(self, station):
        """
        Load a Lasso Regression model for a specific station.
        """
        self.model = joblib.load(os.path.join(self.model_path, f"{station}.joblib"))

    def transform_data_to_predict(self, data):
        """
        Transform the last 30 days of weather data into a 1D array for prediction.
        """
        # Extract previous 30 days of relevant columns
        previous_data = data[['TMIN', 'TAVG', 'TMAX', 'SNOW', 'PRCP']].tail(30).values
        # Flatten the data into a 1D array
        previous_data = previous_data.flatten()
        return previous_data

    def predict(self, data, station):
        """
        Predict using the loaded model for the specified station.
        """
        if isinstance(data, str):
            data = pd.read_csv(data)
        # Load the model for the station
        self.load_model(station)
        # Transform the data into the required format
        X = self.transform_data_to_predict(data)
        X = X.reshape(1, -1)
        # Make predictions
        predictions = self.model.predict(X)
        return predictions.reshape(-1)
class RandomForestPredictor:
    def __init__(self, model_path="predictor/models/RandomForest/"):
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        self.model = None
        self.model_path = os.path.join(self.repo_root, model_path)

    def train_and_save_model(self):
        """
        Train a Random Forest model for each station and save it.
        """
        # Ensure the model path exists
        os.makedirs(self.model_path, exist_ok=True)

        # Hyperparameter grid for Random Forest
        '''
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30]
        }
        '''


        for station in stations_list:
            print(f"Training model for station: {station}")
            station_file_path = os.path.join(self.repo_root, f"analysis/regression_data/{station}.csv")
            # Load the data for the specified station
            station_data = pd.read_csv(station_file_path)
            # The first 153 columns are the features, the last 15 columns are the labels
            X = station_data.iloc[:, 3:153]
            y = station_data.iloc[:, 153:]

            # Initialize the base Random Forest model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            joblib.dump(model, os.path.join(self.model_path, f"{station}.joblib"))


            '''
            # Set up GridSearchCV for hyperparameter tuning with 5-fold CV
             grid_search = GridSearchCV(
                estimator=rf,
                param_grid=param_grid,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=2
            )
            
           

            # Fit the GridSearchCV to find the best model
            grid_search.fit(X, y)

            # Extract the best model
            best_model = grid_search.best_estimator_
            print(f"Best parameters for station {station}: {grid_search.best_params_}")
            # Save the best model
            joblib.dump(best_model, os.path.join(self.model_path, f"{station}.joblib"))
            '''

    def load_model(self, station):
        """
        Load a Random Forest model for a specific station.
        """
        self.model = joblib.load(os.path.join(self.model_path, f"{station}.joblib"))

    def transform_data_to_predict(self, data):
        """
        Transform the last 30 days of weather data into a 1D array for prediction.
        """
        # Extract previous 30 days of relevant columns
        previous_data = data[['TMIN', 'TAVG', 'TMAX', 'SNOW', 'PRCP']].tail(30).values
        # Flatten the data into a 1D array
        previous_data = previous_data.flatten()
        return previous_data

    def predict(self, data, station):
        """
        Predict using the loaded model for the specified station.
        """
        if isinstance(data, str):
            data = pd.read_csv(data)
        # Load the model for the station
        self.load_model(station)
        # Transform the data into the required format
        X = self.transform_data_to_predict(data)
        X = X.reshape(1, -1)
        # Make predictions
        predictions = self.model.predict(X)
        return predictions.reshape(-1)



class XGBoostPredictor:
    def __init__(self, model_path="predictor/models/XGBoost/"):
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        self.model = None
        self.model_path = os.path.join(self.repo_root, model_path)

    def train_and_save_model(self):
        """
        Train an XGBoost model for each station and save it.
        """
        # Ensure the model path exists
        os.makedirs(self.model_path, exist_ok=True)

        for station in stations_list:
            print(f"Training model for station: {station}")
            station_file_path = os.path.join(self.repo_root, f"analysis/regression_data/{station}.csv")

            # Load the data for the specified station
            station_data = pd.read_csv(station_file_path)
            # The first 153 columns are the features, the last 15 columns are the labels
            X = station_data.iloc[:, 3:153]
            y = station_data.iloc[:, 153:]

            # Initialize the XGBoost model
            model = XGBRegressor(objective='reg:squarederror', learning_rate=0.01, max_depth = 6, n_estimators=200, random_state=42)

            # Train the model
            model.fit(X, y)

            # Save the model
            joblib.dump(model, os.path.join(self.model_path, f"{station}.joblib"))

    def load_model(self, station):
        """
        Load an XGBoost model for a specific station.
        """
        self.model = joblib.load(os.path.join(self.model_path, f"{station}.joblib"))

    def transform_data_to_predict(self, data):
        """
        Transform the last 30 days of weather data into a 1D array for prediction.
        """
        # Extract previous 30 days of relevant columns
        previous_data = data[['TMIN', 'TAVG', 'TMAX', 'SNOW', 'PRCP']].tail(30).values
        # Flatten the data into a 1D array
        previous_data = previous_data.flatten()
        return previous_data

    def predict(self, data, station):
        """
        Predict using the loaded model for the specified station.
        """
        if isinstance(data, str):
            data = pd.read_csv(data)
        # Load the model for the station
        self.load_model(station)
        # Transform the data into the required format
        X = self.transform_data_to_predict(data)
        X = X.reshape(1, -1)
        # Make predictions
        predictions = self.model.predict(X)
        return predictions.reshape(-1)
# Train Linear Regression model
#model = RandomForestPredictor()
#model.train_and_save_model("predictor/models/RandomForest/")
#model.predict("../data/restructured_simple/PANC.csv", "PANC")

# summarize the model
# print(model.summary)


# Example usage
#predictor = PreviousDayPredictor("data/processed/openweather_hourly")
# predictor._load_data()
#print(predictor.predict("2023-11-19"))

