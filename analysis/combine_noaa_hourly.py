import pandas as pd
from datetime import datetime


def process_weather_data(data_file, hourly_data_file):
    data = pd.read_csv(data_file)
    hourly_data = pd.read_csv(hourly_data_file)

    # Convert the 'Datetime' column in 'hourly_data' to a datetime object and extract Year, Month, Day
    hourly_data['Datetime'] = pd.to_datetime(hourly_data[['Year', 'Month', 'Day']])
    hourly_data['Year'] = hourly_data['Datetime'].dt.year
    hourly_data['Month'] = hourly_data['Datetime'].dt.month
    hourly_data['Day'] = hourly_data['Datetime'].dt.day

    start_date = pd.to_datetime('2023-11-30')
    end_date = pd.to_datetime(datetime.today().strftime('%Y-%m-%d'))

    data.loc[
        (data['YEAR'] == 2024) &
        ((data['MONTH'] == 11) & (data['DAY'] >= 12)),
        ['TAVG', 'TMIN', 'TMAX', 'PRCP']
    ] = None  # Replace values of these columns to NA for this date range

    hourly_aggregated = hourly_data.groupby(['Year', 'Month', 'Day']).agg(
        TAVG=('Temperature (F)', 'mean'),  # Average Temperature to TAVG
        TMIN=('Temp Min (F)', 'min'),  # Min Temp to TMIN
        TMAX=('Temp Max (F)', 'max'),  # Max Temp to TMAX
        PRCP=('Rain (1h)', 'sum'),  # Pressure to PRCP
        SNOW=('Snow (1h)', 'sum')
    ).reset_index()

    # Rename the columns in hourly_aggregated to avoid conflict during merge
    hourly_aggregated = hourly_aggregated.rename(columns={
        'TAVG': 'TAVG_new',
        'TMIN': 'TMIN_new',
        'TMAX': 'TMAX_new',
        'PRCP': 'PRCP_new',
        'SNOW': 'SNOW_new'
    })

    # Merge the aggregated hourly data with 'data', replacing NA values with hourly aggregated values
    data = pd.merge(data, hourly_aggregated, how='outer', left_on=['YEAR', 'MONTH', 'DAY'],
                    right_on=['Year', 'Month', 'Day'])

    # Replace the NA values in the columns with the values from the aggregated hourly
    data['YEAR'] = data['YEAR'].combine_first(data['Year']).astype(int)
    data['MONTH'] = data['MONTH'].combine_first(data['Month']).astype(int)
    data['DAY'] = data['DAY'].combine_first(data['Day']).astype(int)

    data['TAVG'] = data['TAVG_new'].combine_first(data['TAVG'])
    data['TAVG'] = data['TAVG_new'].combine_first(data['TAVG'])  # Replace NA in 'TAVG' with hourly aggregated value
    data['TMIN'] = data['TMIN_new'].combine_first(data['TMIN'])  # Replace NA in 'TMIN' with hourly aggregated value
    data['TMAX'] = data['TMAX_new'].combine_first(data['TMAX'])  # Replace NA in 'TMAX' with hourly aggregated value
    data['PRCP'] = data['PRCP_new'].combine_first(data['PRCP'])  # Replace NA in 'PRCP' with hourly aggregated value
    data['SNOW'] = data['SNOW_new'].combine_first(data['SNOW'])
    # Drop the extra columns created during the merge (e.g., columns with '_new' suffix)
    data['SNWD'] = data['SNWD'].fillna(0)
    data = data.drop(columns=[col for col in data.columns if col.endswith('_new') or col in ['Year', 'Month', 'Day']])

    return data


import os


def process_all_weather_data(data_directory, hourly_data_directory, locations):
    """
    Process all weather data files in the given directories using the provided location mapping.

    :param data_directory: The directory containing the data CSV files (e.g., ../data/restructured/).
    :param hourly_data_directory: The directory containing the hourly data CSV files (e.g., ../data/processed/openweather_hourly/).
    :param locations: A dictionary of locations mapping city names to their respective airport codes and other info.
    """

    # Ensure the 'combined' directory exists within data_directory
    combined_directory = os.path.join(data_directory, 'combined')
    os.makedirs(combined_directory, exist_ok=True)  # Create the combined directory if it doesn't exist

    # Iterate through each location in the locations dictionary
    for city, details in locations.items():
        airport_id = details['id']
        hourly_data_file = f"{hourly_data_directory}/{city}_hourly_data.csv"
        data_file = f"{data_directory}/{airport_id}.csv"

        if os.path.exists(data_file) and os.path.exists(hourly_data_file):
            print(f"Processing data for {city}...")

            # Process the data using the process_weather_data function
            final_data = process_weather_data(data_file, hourly_data_file)

            # Save the processed data to the 'combined' directory
            output_file = os.path.join(combined_directory, f"{airport_id}_data_combined.csv")
            final_data.to_csv(output_file, index=False)
            print(f"Processed data saved to {output_file}")

        else:
            print(f"Files not found for {city}. Skipping...")


# Example usage
cities = {
    "Anchorage": {"id": "PANC"},
    "Boise": {"id": "KBOI"},
    "Chicago": { "id": "KORD"},
    "Denver": {"id": "KDEN"},
    "Detroit": {"id": "KDTW"},
    "Honolulu": {"id": "PHNL"},
    "Houston": { "id": "KIAH"},
    "Miami": {"id": "KMIA"},
    "Minneapolis": {"id": "KMSP"},
    "Oklahoma City": {"id": "KOKC"},
    "Nashville": {"id": "KBNA"},
    "New York": {"id": "KJFK"},
    "Phoenix": {"id": "KPHX"},
    "Portland ME": {"id": "KPWM"},
    "Portland OR": {"id": "KPDX"},
    "Salt Lake City": {"id": "KSLC"},
    "San Diego": {"id": "KSAN"},
    "San Francisco": {"id": "KSFO"},
    "Seattle": {"id": "KSEA"},
    "Washington DC": {"id": "KDCA"}
}

# Directory paths (these should be adjusted to your specific directory structure)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Directory where the JSON files are stored
data_directory = os.path.join(repo_root, 'data/restructured_simple')
hourly_data_directory = os.path.join(repo_root, 'data/processed/openweather_hourly')

# Process all weather data
process_all_weather_data(data_directory, hourly_data_directory, cities)