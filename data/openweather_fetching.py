import csv
import os
import warnings
warnings.filterwarnings("ignore")
import requests


API_KEY = os.getenv('OPENWEATHER_API_KEY')

# Dictionary containing city names and their coordinates (latitude, longitude)
city_coordinates = {
    'Anchorage': (61.2181, -149.9003),
    'Boise': (43.615, -116.2023),
    'Chicago': (41.8781, -87.6298),
    'Denver': (39.7392, -104.9903),
    'Detroit': (42.3314, -83.0458),
    'Honolulu': (21.3069, -157.8583),
    'Houston': (29.7604, -95.3698),
    'Miami': (25.7617, -80.1918),
    'Minneapolis': (44.9778, -93.265),
    'Oklahoma City': (35.4676, -97.5164),
    'Nashville': (36.1627, -86.7816),
    'New York': (40.7128, -74.006),
    'Phoenix': (33.4484, -112.074),
    'Portland ME': (43.6591, -70.2568),
    'Portland OR': (45.5051, -122.675),
    'Salt Lake City': (40.7608, -111.891),
    'San Diego': (32.7157, -117.1611),
    'San Francisco': (37.7749, -122.4194),
    'Seattle': (47.6062, -122.3321),
    'Washington DC': (38.9072, -77.0369)
}

BASE_URL = 'https://api.openweathermap.org/data/2.5/weather'

# Directory to save the combined weather data file
output_dir = 'openweather/current'
os.makedirs(output_dir, exist_ok=True)

# CSV file path
csv_file_path = os.path.join(output_dir, "detailed_weather_data.csv")

# Fetch weather data and save to CSV
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow([
        "City", "Latitude", "Longitude", "Weather Description", "Temperature (F)", "Feels Like (F)",
        "Temp Min (F)", "Temp Max (F)", "Pressure", "Humidity", "Sea Level", "Ground Level",
        "Wind Speed", "Wind Deg", "Wind Gust", "Clouds All", "Datetime", "Country",
        "Sunrise", "Sunset", "Timezone", "City ID", "City Name"
    ])

    # Function to fetch weather data for a city
    def fetch_weather(city, lat, lon):
        params = {
            'lat': lat,
            'lon': lon,
            'appid': API_KEY,
            'units': 'imperial'  # Use 'metric' for Celsius
        }
        response = requests.get(BASE_URL, params=params)
        if response.status_code == 200:
            weather_data = response.json()
            # Extract relevant data
            coord = weather_data.get('coord', {})
            weather = weather_data.get('weather', [{}])[0]
            main = weather_data.get('main', {})
            wind = weather_data.get('wind', {})
            clouds = weather_data.get('clouds', {})
            sys = weather_data.get('sys', {})

            # Create a row with all requested data, handling missing values
            return [
                city,
                coord.get('lat', 'N/A'), coord.get('lon', 'N/A'),
                weather.get('description', 'N/A'),
                main.get('temp', 'N/A'), main.get('feels_like', 'N/A'),
                main.get('temp_min', 'N/A'), main.get('temp_max', 'N/A'),
                main.get('pressure', 'N/A'), main.get('humidity', 'N/A'),
                main.get('sea_level', 'N/A'), main.get('grnd_level', 'N/A'),
                wind.get('speed', 'N/A'), wind.get('deg', 'N/A'), wind.get('gust', 'N/A'),
                clouds.get('all', 'N/A'),
                weather_data.get('dt', 'N/A'), sys.get('country', 'N/A'),
                sys.get('sunrise', 'N/A'), sys.get('sunset', 'N/A'),
                weather_data.get('timezone', 'N/A'), weather_data.get('id', 'N/A'),
                weather_data.get('name', 'N/A')
            ]
        else:
            print(f"Error fetching data for {city}. Status code: {response.status_code}")
            return [city] + ["Error"] * 21

    # Fetch and write weather data for each city
    for city, (lat, lon) in city_coordinates.items():
        city_weather = fetch_weather(city, lat, lon)
        writer.writerow(city_weather)

print(f"Detailed weather data saved to {csv_file_path}")
