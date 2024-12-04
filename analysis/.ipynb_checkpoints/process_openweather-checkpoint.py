import json
import os
import pandas as pd
from datetime import datetime
import pytz

# Define a dictionary to map city names to their respective time zones
city_timezones = {
    'Anchorage': 'America/Anchorage',
    'Boise': 'America/Boise',
    'Chicago': 'America/Chicago',
    'Denver': 'America/Denver',
    'Detroit': 'America/Detroit',
    'Honolulu': 'Pacific/Honolulu',
    'Houston': 'America/Chicago',
    'Miami': 'America/New_York',
    'Minneapolis': 'America/Chicago',
    'Oklahoma City': 'America/Chicago',
    'Nashville': 'America/Chicago',
    'New York': 'America/New_York',
    'Phoenix': 'America/Phoenix',
    'Portland ME': 'America/New_York',
    'Portland OR': 'America/Los_Angeles',
    'Salt Lake City': 'America/Denver',
    'San Diego': 'America/Los_Angeles',
    'San Francisco': 'America/Los_Angeles',
    'Seattle': 'America/Los_Angeles',
    'Washington DC': 'America/New_York'
}

# Function to process and save data to CSV
def process_and_save_data(data_json, city_name):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Initialize an empty list to store the rows of data
    rows = []
    timezone = pytz.timezone(city_timezones[city_name])  # Get the timezone for the city

    # Loop through each entry in the "list" field of the JSON data
    for chunk in data_json:
        for entry in chunk:
            # Extract the Unix timestamp and convert to local datetime
            dt_utc = datetime.utcfromtimestamp(entry['dt']).replace(tzinfo=pytz.utc)
            dt_local = dt_utc.astimezone(timezone)

            # Extract the components: year, month, day, time
            year = dt_local.year
            month = dt_local.month
            day = dt_local.day
            time = dt_local.strftime('%H:%M:%S')  # Time in HH:MM:SS format

            # Extract temperature, which is already in Fahrenheit
            temp = entry['main']['temp']
            feels_like = entry['main']['feels_like']
            temp_min = entry['main']['temp_min']
            temp_max = entry['main']['temp_max']

            # Extract pressure and humidity
            pressure = entry['main']['pressure']
            humidity = entry['main']['humidity']

            # Extract wind details
            wind_speed = entry['wind']['speed']
            wind_deg = entry['wind']['deg']
            wind_gust = entry['wind'].get('gust', None)

            # Extract cloud coverage
            clouds_all = entry['clouds']['all']

            # Extract rain and snow data, default to 0 if not present
            rain = entry.get('rain', {}).get('1h', 0)
            snow = entry.get('snow', {}).get('1h', 0)
            # Extract weather description
            weather_description = entry['weather'][0]['description']

            # Append the data as a row
            rows.append([
                year, month, day, time, temp, feels_like, temp_min, temp_max,
                pressure, humidity, wind_speed, wind_deg, wind_gust, clouds_all,
                rain, snow, weather_description
            ])

    # Create a DataFrame with the specified column names
    df = pd.DataFrame(rows, columns=[
        'Year', 'Month', 'Day', 'Time', 'Temperature (F)', 'Feels Like (F)',
        'Temp Min (F)', 'Temp Max (F)', 'Pressure', 'Humidity', 'Wind Speed',
        'Wind Deg', 'Wind Gust', 'Clouds All', 'Rain (1h)', 'Snow (1h)', 'Weather_description'
    ])

    # Ensure the directory exists
    directory = os.path.join(repo_root, 'data/processed/openweather_hourly')
    os.makedirs(directory, exist_ok=True)

    # Define the file path for saving the CSV
    file_path = os.path.join(directory, f'{city_name}_hourly_data.csv')

    # Save the DataFrame to a CSV file
    df.to_csv(file_path, index=False)
    # print(f"Data saved to '{file_path}'.")


# Function to process all JSON files and convert them to CSV
def process_all_json_files(json_directory):
    # Get the list of all JSON files in the given directory
    json_files = [f for f in os.listdir(json_directory) if f.endswith('_hourly_data.json')]

    # Process each JSON file
    for json_file in json_files:
        city_name = json_file.split('_')[0]  # Extract city name from the filename
        json_path = os.path.join(json_directory, json_file)

        # Load the JSON data
        with open(json_path, 'r') as file:
            data_json = json.load(file)

        # Process and save the data as CSV
        if city_name in city_timezones:
            process_and_save_data(data_json, city_name)
        else:
            print(f"Timezone for city '{city_name}' not found. Skipping.")

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Directory where the JSON files are stored
json_directory = os.path.join(repo_root, 'data/original/openweather_hourly')
# Call the function to process all JSON files in the specified directory
process_all_json_files(json_directory)
