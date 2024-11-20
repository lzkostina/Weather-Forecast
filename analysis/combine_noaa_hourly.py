import pandas as pd
from datetime import datetime

def process_weather_data(data_file, hourly_data_file):
    # Step 1: Load the CSV files
    data = pd.read_csv(data_file)
    hourly_data = pd.read_csv(hourly_data_file)

    # Step 2: Preprocess the hourly data to ensure proper date format
    # Convert the 'Datetime' column in 'hourly_data' to a datetime object and extract Year, Month, Day
    hourly_data['Datetime'] = pd.to_datetime(hourly_data[['Year', 'Month', 'Day']])
    hourly_data['Year'] = hourly_data['Datetime'].dt.year
    hourly_data['Month'] = hourly_data['Datetime'].dt.month
    hourly_data['Day'] = hourly_data['Datetime'].dt.day

    # Step 3: Define the date range for filtering
    start_date = pd.to_datetime('2023-11-30')
    end_date = pd.to_datetime(datetime.today().strftime('%Y-%m-%d'))

    # Step 4: Replace all feature columns in the date range (2023-11-30 to today) with NA
    data.loc[
        (data['YEAR'] == 2023) &
        ((data['MONTH'] == 11) & (data['DAY'] >= 30)) |
        (data['YEAR'] > 2023),
        ['TAVG', 'TMIN', 'TMAX', 'PRCP', 'RHAV', 'WSF1']
    ] = None  # Replace values of these columns to NA for this date range

    # Step 5: Aggregate the 'hourly_data' by day and rename columns to match 'data'
    hourly_aggregated = hourly_data.groupby(['Year', 'Month', 'Day']).agg(
        TAVG=('Temperature (F)', 'mean'),  # Average Temperature to TAVG
        TMIN=('Temp Min (F)', 'min'),      # Min Temp to TMIN
        TMAX=('Temp Max (F)', 'max'),      # Max Temp to TMAX
        PRCP=('Pressure', 'mean'),        # Pressure to PRCP
        RHAV=('Humidity', 'mean'),        # Humidity to RHAV
        WSF1=('Wind Speed', 'mean')       # Wind Speed to WSF1
    ).reset_index()

    # Step 6: Rename the columns in hourly_aggregated to avoid conflict during merge
    hourly_aggregated = hourly_aggregated.rename(columns={
        'TAVG': 'TAVG_new',
        'TMIN': 'TMIN_new',
        'TMAX': 'TMAX_new',
        'PRCP': 'PRCP_new',
        'RHAV': 'RHAV_new',
        'WSF1': 'WSF1_new'
    })

    # Step 7: Merge the aggregated hourly data with 'data', replacing NA values with hourly aggregated values
    data = pd.merge(data, hourly_aggregated, how='left', left_on=['YEAR', 'MONTH', 'DAY'], right_on=['Year', 'Month', 'Day'])

    # Step 8: Replace the NA values in the columns with the values from the aggregated hourly data
    data['TAVG'] = data['TAVG_new'].combine_first(data['TAVG'])  # Replace NA in 'TAVG' with hourly aggregated value
    data['TMIN'] = data['TMIN_new'].combine_first(data['TMIN'])  # Replace NA in 'TMIN' with hourly aggregated value
    data['TMAX'] = data['TMAX_new'].combine_first(data['TMAX'])  # Replace NA in 'TMAX' with hourly aggregated value
    data['PRCP'] = data['PRCP_new'].combine_first(data['PRCP'])  # Replace NA in 'PRCP' with hourly aggregated value
    data['RHAV'] = data['RHAV_new'].combine_first(data['RHAV'])  # Replace NA in 'RHAV' with hourly aggregated value
    data['WSF1'] = data['WSF1_new'].combine_first(data['WSF1'])  # Replace NA in 'WSF1' with hourly aggregated value

    # Step 9: Drop the extra columns created during the merge (e.g., columns with '_new' suffix)
    data = data.drop(columns=[col for col in data.columns if col.endswith('_new') or col in ['Year', 'Month', 'Day']])

    # Step 10: Return the final merged dataset
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
            output_file = os.path.join(combined_directory, f"processed_{airport_id}_data.csv")
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
data_directory = "../data/restructured"
hourly_data_directory = "../data/processed/openweather_hourly"

# Process all weather data
process_all_weather_data(data_directory, hourly_data_directory, cities)