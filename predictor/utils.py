import os
import urllib.request
import logging

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


