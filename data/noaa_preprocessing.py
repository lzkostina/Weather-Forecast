import os
import urllib.request
import logging

station_code_dict = {
    "PANC": "USW00026451", # Anchorage 
    "KBOI": "USW00024131", # Boise  
    "KORD": "USW00094846", # Chicago
    "KDEN": "USW00003017", # Denver 
    "KDTW": "USW00094847", # Detroit
    "PHNL": "USW00022521", # Honolulu 
    "KIAH": "USW00012960", # Houston
    "KMIA": "USW00012839", # Miami 
    "KMSP": "USW00014922", # Minneapolis 
    "KOKC": "USW00013967", # Oklahoma City 
    "KBNA": "USW00013897", # Nashville 
    "KJFK": "USW00094789", # New York 
    "KPHX": "USW00023183", # Phoenix 
    "KPWM": "USW00014764", # Portland ME
    "KPDX": "USW00024229", # Portland OR 
    "KSLC": "USW00024127", # Salt Lake City
    "KSAN": "USW00023188", # San Diego 
    "KSFO": "USW00023234", # San Francisco 
    "KSEA": "USW00024233", # Seattle 
    "KDCA": "USW00013743", # Washington DC
}

data_path_url = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/"

# Directory to save downloaded files
raw_noaa_cache = "data/"

# Ensure the directory exists
os.makedirs(raw_noaa_cache, exist_ok=True)

# URL to download data from
data_path_url = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/"

# Setup logging
logging.basicConfig(level=logging.INFO)

# Loop through the station codes and download the corresponding files
for station_code, file_name in station_code_dict.items():
    url = f"{data_path_url}{file_name}.dly"
    try:
        # Download the file and save it
        urllib.request.urlretrieve(url, os.path.join(raw_noaa_cache, f"{station_code}.dly"))
        logging.info(f"Successfully scraped data for: {station_code}")
    except Exception as e:
        logging.error(f"Failed to download data for {station_code}: {e}")



