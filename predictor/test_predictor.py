import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from abc import ABC, abstractmethod

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

class Predictor(ABC): 
    @abstractmethod
    def predict(self, data):
        """
        Abstract method to be implemented by subclasses to perform predictions.

        Predict function should return a list of 300 numbers in an np array.
        """
        pass

class TestPredictor(Predictor):
    def __init__(self):
        pass

    def predict(self, data):

        
        # Generate 300 numbers of the form "XX.X" (all zeroes with one decimal place)
        predictions = np.zeros(300)
        #predictions_rounded = np.around(predictions, 1)
        
        # Create the output string
        #output = f'"{current_date}" is the current date and then {", ".join(predictions_rounded)}'
        
        return predictions

class PreviousDayPredictor(Predictor):
    def __init__(self, input_directory= "data/processed/openweather_hourly"):
        self.input_directory = input_directory
        self.data = self._load_data()

    def _load_data(self):
        """
        Loads all CSV files from the input directory and concatenates them into a single DataFrame.
        """
        all_files = [os.path.join(self.input_directory, f) for f in os.listdir(self.input_directory) if f.endswith('.csv')]
        dataframes = []
        for file in all_files:
            location_name = os.path.basename(file).split('_')[0]
            df = pd.read_csv(file)
            df['Location'] = location_name
            dataframes.append(df)
        combined_data = pd.concat(dataframes, ignore_index=True)
        combined_data['Datetime'] = pd.to_datetime(
            combined_data[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1) + ' ' + combined_data['Time']
        )
        combined_data.sort_values(by='Datetime', inplace=True)
        return combined_data

    def predict(self, start_date):
        """
        Predicts temperature for the next 5 days using the previous day's min, avg, and max temperature.

        Args:
            start_date (str): The start date for prediction in 'YYYY-MM-DD' format.

        Returns:
            str: A formatted string with the current date and 300 predictions in the format XX.X.
        """
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        previous_day = start_date - timedelta(days=1)

        # List of locations
        locations = [
            "Anchorage", "Boise", "Chicago", "Denver", "Detroit", "Honolulu", "Houston", "Miami", "Minneapolis",
            "Oklahoma City", "Nashville", "New York", "Phoenix", "Portland ME", "Portland OR", "Salt Lake City",
            "San Diego", "San Francisco", "Seattle", "Washington DC"
        ]

        predictions = []
        for location in locations:
            # Filter data for the previous day and specific location
            previous_day_data = self.data[(self.data['Datetime'].dt.date == previous_day.date()) & (self.data['Location'] == location)]

            if previous_day_data.empty:
                raise ValueError(f"No data available for the previous day: {previous_day.strftime('%Y-%m-%d')} for location: {location}")

            # Calculate min, avg, and max temperature for the previous day
            min_temp = previous_day_data['Temperature (F)'].min()
            avg_temp = previous_day_data['Temperature (F)'].mean()
            max_temp = previous_day_data['Temperature (F)'].max()

            # Prepare predictions for the next 5 days for each location
            for _ in range(5):
                predictions.extend([min_temp, avg_temp, max_temp])

        return predictions



# Example usage
#predictor = PreviousDayPredictor("data/processed/openweather_hourly")
# predictor._load_data()
#print(predictor.predict("2023-11-19"))

# previous_day_data = predictor.data[(predictor.data['Datetime'].dt.date == previous_day.date()) & (predictor.data['Location'] == location)]

