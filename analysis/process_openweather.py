import json
import os
import pandas as pd
from datetime import datetime


# Function to process and save data to CSV
def process_and_save_data(data_json, city_name):
    # Initialize an empty list to store the rows of data
    rows = []

    # Loop through each entry in the "list" field of the JSON data
    for chunk in data_json:
        for entry in chunk:
            # Extract the Unix timestamp and convert to datetime
            dt = datetime.utcfromtimestamp(entry['dt'])

            # Extract the components: year, month, day, time
            year = dt.year
            month = dt.month
            day = dt.day
            time = dt.strftime('%H:%M:%S')  # Time in HH:MM:SS format

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

            # Extract weather description
            weather_description = entry['weather'][0]['description']

            # Append the data as a row
            rows.append([year, month, day, time, temp, feels_like, temp_min, temp_max, pressure, humidity, wind_speed,
                         wind_deg, wind_gust, clouds_all, weather_description])

    # Create a DataFrame with the specified column names
    df = pd.DataFrame(rows, columns=['Year', 'Month', 'Day', 'Time', 'Temperature (F)', 'Feels Like (F)',
                                     'Temp Min (F)', 'Temp Max (F)', 'Pressure', 'Humidity', 'Wind Speed',
                                     'Wind Deg', 'Wind Gust', 'Clouds All', 'Weather Description'])

    # Ensure the directory exists
    directory = '../data/processed/openweather_hourly'
    os.makedirs(directory, exist_ok=True)

    # Define the file path for saving the CSV
    file_path = os.path.join(directory, f'{city_name}_hourly_data.csv')

    # Save the DataFrame to a CSV file
    df.to_csv(file_path, index=False)
    print(f"Data saved to '{file_path}'.")


# Function to process all JSON files and convert them to CSV
def process_all_json_files(json_directory):
    # Get the list of all JSON files in the given directory
    json_files = [f for f in os.listdir(json_directory) if f.endswith('_hourly_data.json')]

    # Process each JSON file
    for json_file in json_files:
        city_name = json_file.split('_')[
            0]  # Extract city name from the filename (assuming format 'city_name_hourly_data_fahrenheit.json')
        json_path = os.path.join(json_directory, json_file)

        # Load the JSON data
        with open(json_path, 'r') as file:
            data_json = json.load(file)

        # Process and save the data as CSV
        process_and_save_data(data_json, city_name)


# Directory where the JSON files are stored
json_directory = '../data/original/openweather_hourly'

# Call the function to process all JSON files in the specified directory
process_all_json_files(json_directory)
