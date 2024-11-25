import pandas as pd
import numpy as np
import os
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

def restructure_climate_data(input_csv, output_csv):
    # Load the dataset
    data = pd.read_csv(input_csv, low_memory=False)

    # Reshape the data
    all_days = []

    def process_row(row):
        for day in range(1, 32):  # Loop through each day in the month
            day_data = {
                'YEAR': row['YEAR'],
                'MONTH': row['MONTH'],
                'DAY': day
            }
            # Add the element's value for the day if present
            value_col = f'VALUE{day}'

            day_data[row['ELEMENT']] = float(row[value_col]) if pd.notna(row[value_col]) else np.nan
            all_days.append(day_data)

    # Apply processing for all rows in the dataset
    data.apply(process_row, axis=1)

    # Combine all processed rows into a DataFrame
    expanded_data = pd.DataFrame(all_days)
    # Pivot data to ensure all elements are included as separate columns
    final_data = expanded_data.pivot_table(
        index=['YEAR', 'MONTH', 'DAY'],
        values=[col for col in expanded_data.columns if col not in ['YEAR', 'MONTH', 'DAY']],
        aggfunc='first'
    ).reset_index()

    # Convert temperatures from tenths of degrees Celsius to Fahrenheit
    temp_columns = ['TMAX', 'TMIN', 'TAVG']
    for col in temp_columns:
        if col in final_data.columns:
            final_data[col] = final_data[col].apply(lambda x: (x / 10 * 9 / 5 + 32)  if pd.notna(x) else x)

    # Remove days that are invalid
    final_data = final_data[
        final_data.apply(lambda x: is_valid_date(int(x['YEAR']), int(x['MONTH']), int(x['DAY'])), axis=1)]


    # Filter out columns that have 4000 or more missing values for the years 2014-2024
    # filtered_data = final_data[(final_data['YEAR'] >= 2014) & (final_data['YEAR'] <= 2024)]
    # missing_values_count_filtered = filtered_data.isna().sum()
    # columns_to_drop = missing_values_count_filtered[missing_values_count_filtered >= 4000].index
    # final_data = final_data.drop(columns=columns_to_drop)

    # If missing value for TAVG, calculate the average of TMAX and TMIN
    final_data['TAVG'] = final_data.apply(
        lambda x: (x['TMAX'] + x['TMIN']) / 2 if pd.isna(x['TAVG']) else x['TAVG'], axis=1)

    # If missing value for SNOW or SNWD, fill with 0
    final_data['SNOW'] = final_data['SNOW'].fillna(0)
    final_data['SNWD'] = final_data['SNWD'].fillna(0)

    final_data = final_data.reset_index(drop=True)

    # If TMIN or TMAX is <-1000, use the previous days value
    def replace_invalid_temperatures(data):
        for i in range(1, len(data)):
            if data.loc[i, 'TMIN'] < -1000 and pd.notna(data.loc[i - 1, 'TMIN']):
                data.loc[i, 'TMIN'] = data.loc[i - 1, 'TMIN']
            if data.loc[i, 'TMAX'] < -1000 and pd.notna(data.loc[i - 1, 'TMAX']):
                data.loc[i, 'TMAX'] = data.loc[i - 1, 'TMAX']
        return data

    # Apply the replacement function
    final_data = final_data.sort_values(by=['YEAR', 'MONTH', 'DAY'])  # Ensure data is sorted chronologically
    final_data = replace_invalid_temperatures(final_data)

    # If SNWD, SNOW, or PRCP is <0, use 0
    final_data['SNWD'] = final_data['SNWD'].apply(lambda x: 0 if x < 0 else x)
    final_data['SNOW'] = final_data['SNOW'].apply(lambda x: 0 if x < 0 else x)
    final_data['PRCP'] = final_data['PRCP'].apply(lambda x: 0 if x < 0 else x)


    # Save the restructured data to a new CSV file
    final_data.to_csv(output_csv, index=False)

    print(f"Data has been restructured and saved to '{output_csv}'.")


def count_missing_values(output_csv):
    # Load the restructured dataset
    final_data = pd.read_csv(output_csv)

    # 1) Count rows with missing values for each column
    #missing_values_count = final_data.isna().sum().sort_values()
    #print("Missing values for each column (sorted low to high):")
    #pd.set_option('display.max_rows', None)
    #print(missing_values_count)

    # 2) Count rows with missing values for each column for the years 2014-2024
    filtered_data = final_data[(final_data['YEAR'] >= 2014) & (final_data['YEAR'] <= 2023)]
    missing_values_count_filtered = filtered_data.isna().sum().sort_values()
    print("Missing values for each column (2014-2024, sorted low to high):")
    print(missing_values_count_filtered)

# Print missing values for all stations
def count_missing_values_all_stations(input_directory):
    # Process all CSV files in the specified directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.csv') and len(filename) == 8:
            input_csv = os.path.join(input_directory, filename)
            print(f"Counting missing values for file: {filename}")
            count_missing_values(input_csv)

    print("Missing value counting complete for all files in the directory.")

# count_missing_values_all_stations('data/restructured_simple')

# function to count values < -1000 for each column
def count_values_below_threshold(output_csv, threshold):
    # Load the restructured dataset
    final_data = pd.read_csv(output_csv)
    filtered_data = final_data[(final_data['YEAR'] >= 2014) & (final_data['YEAR'] <= 2023)]

    # Count rows with values below the threshold for each column
    below_threshold_count = final_data[filtered_data < threshold].count()
    print(f"Values below {threshold} for each column:")
    print(below_threshold_count)

# count_values_below_threshold('data/restructured_simple/PHNL.csv', -1000)


def create_core_elements_dataset(output_csv, core_output_csv):
    # Load the restructured dataset
    final_data = pd.read_csv(output_csv)

    # Filter to keep only year, month, day, and the core elements (TMAX, TMIN, PRCP, SNOW, SNWD, TAVG)
    core_elements = ['YEAR', 'MONTH', 'DAY', 'TMAX', 'TMIN', 'PRCP', 'SNOW', 'SNWD', 'TAVG']
    core_data = final_data[core_elements]

    # Save the core elements dataset to a new CSV file
    core_data.to_csv(core_output_csv, index=False)

    print(f"Core elements data has been saved to '{core_output_csv}'.")

def process_all_files_in_directory(input_directory, output_directory, core_output_directory):
    # Process all CSV files in the specified directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.csv') and len(filename) == 8:
            input_csv = os.path.join(input_directory, filename)
            output_csv = os.path.join(output_directory, filename)
            core_output_csv = os.path.join(core_output_directory, filename)
            print(f"Processing file: {filename}")

            # Run restructure and core elements dataset creation
            restructure_climate_data(input_csv, output_csv)
            create_core_elements_dataset(output_csv, core_output_csv)

    print("Processing complete for all files in the directory.")

process_all_files_in_directory('data/processed',
                               'data/restructured',
                               'data/restructured_simple')

