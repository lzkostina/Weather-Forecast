import os
import urllib.request
import logging
import datetime
import pandas as pd
import json



# paths
raw_noaa_cache = "data"
processed_noaa_cache = "data/processed"

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

def load_data(data_folder):
    """
    Need to add annotation here 
    """
    if data_src not in ["noaa"]:
        logging.warning(f"Invalid data source requested: {data_src} -- must be noaa")
        return None

    station_to_processed_data = {}
    for station in stations_list:
        if data_src == "noaa":
            data_src_path = os.path.join(processed_noaa_cache, f"{station}.csv")
        else:
            data_src_path = os.path.join(processed_wunderground_cache, f"{station}.csv") # need to be changed!!!!!
        station_to_processed_data[station] = pd.read_csv(data_src_path, index_col=0)
        station_to_processed_data[station].index = pd.to_datetime(station_to_processed_data[station].index)
    return station_to_processed_data


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


def adjust_predictions(predictions):
    """
    Adjust the predictions to ensure:
    - TMIN < TAVG < TMAX for each day
    """
    # Ensure the predictions can be divided into groups of 3
    if len(predictions) % 3 != 0:
        raise ValueError("The predictions length must be a multiple of 3.")

    # Split predictions into days
    adjusted_predictions = []
    for i in range(0, len(predictions), 3):
        tmin, tavg, tmax = predictions[i:i+3]

        # Adjust TMIN and TMAX to ensure TMIN < TMAX
        tmin, tmax = min(tmin, tavg,tmax), max(tmin, tavg, tmax)

        # Adjust TAVG to be between TMIN and TMAX
        tavg = max(tmin, min(tavg, tmax))

        # Append adjusted values
        adjusted_predictions.extend([tmin, tavg, tmax])

    return adjusted_predictions