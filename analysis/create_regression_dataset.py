# In this file, we will create a regression database to train predictors on.
# Each row of the database will represent one "training example" which consists of:
# - The current date
# - The station string
# - The previous 30 days data of TMIN, TAVG, TMAX, SNOW, PRCP, flattened into a 1D array
# - The labels, which next 5 days data of TMIN, TAVG, TMAX, flattened into a 1D array

import numpy as np
import pandas as pd
import os
import sys
import datetime

def is_valid_date(year, month, day):
  """
  Checks if a given date is valid.

  Args:
    year (int): The year.
    month (int): The month (1-12).
    day (int): The day.

  Returns:
    bool: True if the date is valid, False otherwise.
  """

  try:
    datetime.date(year, month, day)
    return True
  except ValueError:
    return False

stations_list = [
    "PANC", # Anchorage
    "KBOI", # Boise
    "KORD", # Chicago
    "KDEN", # Denver
    "KDTW", # Detroit
    "PHNL", # Honolulu
    "KIAH", # Houston
    "KMIA", # Miami
    "KMSP", # Minneapolis
    "KOKC", # Oklahoma City
    "KBNA", # Nashville
    "KJFK", # New York
    "KPHX", # Phoenix
    "KPWM", # Portland ME
    "KPDX", # Portland OR
    "KSLC", # Salt Lake City
    "KSAN", # San Diego
    "KSFO", # San Francisco
    "KSEA", # Seattle
    "KDCA", # Washington DC
]

# Write function that takes in a given day and station, and returns a single row of the regression dataset

def create_regression_example(year, month, day, station):
    """
    Create a single row of the regression dataset for a given day and station.

    Args:
        year (int): The year of the date for which to create the regression example.
        month (int): The month of the date for which to create the regression example.
        day (int): The day of the date for which to create the regression example.
        station (str): The station for which to create the regression example.

    Returns:
        np.array: A single row of the regression dataset.
    """
    # Get the data for the specified station and year
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    file_path = os.path.join(repo_root, f"data/restructured_simple/{station}.csv")
    df = pd.read_csv(file_path)

    # Find current day index
    current_day_index = df[(df['YEAR'] == year) & (df['MONTH'] == month) & (df['DAY'] == day)].index[0]

    # Get the previous 30 days data of TMIN, TAVG, TMAX, SNOW, PRCP, flattened into a 1D array
    previous_data = df.loc[current_day_index-30:current_day_index-1, ['TMIN', 'TAVG', 'TMAX', 'SNOW', 'PRCP']].values
    previous_data = previous_data.flatten()

    # Get the labels, which are the next 5 days data of TMIN, TAVG, TMAX, flattened into a 1D array
    labels = df.loc[current_day_index:current_day_index+4, ['TMIN', 'TAVG', 'TMAX']].values
    labels = labels.flatten()

    # Combine all the data into a single row
    example = np.concatenate([np.array([year, month, day]), previous_data, labels])

    # reshape the array to be (x,1)
    #example = example.reshape(-1, 1)

    return example





# Create the regression dataset for a given station
# Rules for which current days to include:
# - Only use data from 2014 to 2023
# - Use all days not including November 15 to December 15
# Input: station (string)
# Output: dataframe of regression dataset

def create_regression_dataset(station):

    """
    Create the regression dataset for a given station.

    Args:
        station (str): The station for which to create the regression dataset.

    Returns:
        pd.DataFrame: The regression dataset for the specified station.
    """
    # Initialize an empty list to store the regression examples
    regression_examples = []
    # Loop through all days from 2014 to 2023
    for year in range(2014, 2025):
        for month in range(1, 13):
            for day in range(1, 32):

                if year == 2024 and month >= 11:
                    continue
                # Skip days from November 15 to December 15
                if month == 11 and day >= 15 and day <= 30:
                    continue
                if month == 12 and day >= 1 and day <= 15:
                    continue
                # Skip invalid dates
                if not is_valid_date(year, month, day):
                    continue

                # Create a regression example for the current day
                example = create_regression_example(year, month, day, station)
                regression_examples.append(example)

    # Save the regression examples in a CSV
    regression_df = pd.DataFrame(regression_examples)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    path_dir = os.path.join(repo_root, f'analysis/regression_data/{station}.csv')
    regression_df.to_csv(path_dir, index=False)



def create_regression_dataset_full(station):

    """
    Create the regression dataset for a given station.

    Args:
        station (str): The station for which to create the regression dataset.

    Returns:
        pd.DataFrame: The regression dataset for the specified station.
    """
    # Initialize an empty list to store the regression examples
    regression_examples = []
    # Loop through all days from 2014 to 2023
    for year in range(2014, 2025):
        for month in range(1, 13):
            for day in range(1, 32):
                # Skip invalid dates
                if year == 2024 and month >= 11:
                    continue

                if not is_valid_date(year, month, day):
                    continue

                # Create a regression example for the current day
                example = create_regression_example(year, month, day, station)
                regression_examples.append(example)

    # Save the regression examples in a CSV
    regression_df = pd.DataFrame(regression_examples)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    path_dir = os.path.join(repo_root, f'analysis/regression_data_full/{station}.csv')
    regression_df.to_csv(path_dir, index=False)

# do the same for all stations
for station in stations_list:
    create_regression_dataset(station)
    create_regression_dataset_full(station)
    print(f"Created regression dataset for station {station}")