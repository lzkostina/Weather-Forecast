import requests
import time
import json
import os  # Import os to manage file paths

# Your OpenWeatherMap API Key
API_KEY = 'd817ee55d93f6b51d9b15bc850821726'

# Coordinates for each location
locations = {
    "Anchorage": {"lat": 61.2181, "lon": -149.9003, "id": "PANC"},
    "Boise": {"lat": 43.6150, "lon": -116.2023, "id": "KBOI"},
    "Chicago": {"lat": 41.8781, "lon": -87.6298, "id": "KORD"},
    "Denver": {"lat": 39.7392, "lon": -104.9903, "id": "KDEN"},
    "Detroit": {"lat": 42.3314, "lon": -83.0458, "id": "KDTW"},
    "Honolulu": {"lat": 21.3069, "lon": -157.8583, "id": "PHNL"},
    "Houston": {"lat": 29.7604, "lon": -95.3698, "id": "KIAH"},
    "Miami": {"lat": 25.7617, "lon": -80.1918, "id": "KMIA"},
    "Minneapolis": {"lat": 44.9778, "lon": -93.2650, "id": "KMSP"},
    "Oklahoma City": {"lat": 35.4676, "lon": -97.5164, "id": "KOKC"},
    "Nashville": {"lat": 36.1627, "lon": -86.7816, "id": "KBNA"},
    "New York": {"lat": 40.7128, "lon": -74.0060, "id": "KJFK"},
    "Phoenix": {"lat": 33.4484, "lon": -112.0740, "id": "KPHX"},
    "Portland ME": {"lat": 43.6591, "lon": -70.2568, "id": "KPWM"},
    "Portland OR": {"lat": 45.5051, "lon": -122.6750, "id": "KPDX"},
    "Salt Lake City": {"lat": 40.7608, "lon": -111.8910, "id": "KSLC"},
    "San Diego": {"lat": 32.7157, "lon": -117.1611, "id": "KSAN"},
    "San Francisco": {"lat": 37.7749, "lon": -122.4194, "id": "KSFO"},
    "Seattle": {"lat": 47.6062, "lon": -122.3321, "id": "KSEA"},
    "Washington DC": {"lat": 38.9072, "lon": -77.0369, "id": "KDCA"}
}

# Define the time range: past 1 year (in Unix timestamps)
end_time = int(time.time())  # Current time in Unix timestamp
start_time = end_time - (364 * 24 * 3600)  # One year ago in Unix timestamp

# Directory where data will be saved
directory = '../data/original/openweather_hourly/'

# Ensure the directory exists
os.makedirs(directory, exist_ok=True)

# Loop through each location and retrieve the historical weather data
for city, coords in locations.items():
    lat = coords["lat"]
    lon = coords["lon"]

    # Construct the URL for the API call with units=imperial for Fahrenheit
    url = f'https://history.openweathermap.org/data/2.5/history/city?lat={lat}&lon={lon}&type=hour&start={start_time}&end={end_time}&units=imperial&appid={API_KEY}'

    # Send the API request
    response = requests.get(url)
    data_json = response.json()

    # Check if the request was successful
    if response.status_code == 200 and 'list' in data_json:
        print(f"Data retrieval successful for {city}!")
        # Save the data as a JSON file for later use
        file_name = f'{city}_hourly_data.json'
        file_path = os.path.join(directory, file_name)

        with open(file_path, 'w') as json_file:
            json.dump(data_json, json_file, indent=4)
        print(f"Data saved to '{file_path}'.")
    else:
        print(f"Failed to retrieve data for {city}. Response: {data_json}")
