import pandas as pd
import numpy as np
import os


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

    # Filter out columns that have 4000 or more missing values for the years 2014-2024
    filtered_data = final_data[(final_data['YEAR'] >= 2014) & (final_data['YEAR'] <= 2024)]
    missing_values_count_filtered = filtered_data.isna().sum()
    columns_to_drop = missing_values_count_filtered[missing_values_count_filtered >= 4000].index
    final_data = final_data.drop(columns=columns_to_drop)

    # Save the restructured data to a new CSV file
    final_data.to_csv(output_csv, index=False)

    print(f"Data has been restructured and saved to '{output_csv}'.")


def count_missing_values(output_csv):
    # Load the restructured dataset
    final_data = pd.read_csv(output_csv)

    # 1) Count rows with missing values for each column
    missing_values_count = final_data.isna().sum().sort_values()
    print("Missing values for each column (sorted low to high):")
    pd.set_option('display.max_rows', None)
    print(missing_values_count)

    # 2) Count rows with missing values for each column for the years 2014-2024
    filtered_data = final_data[(final_data['YEAR'] >= 2014) & (final_data['YEAR'] <= 2024)]
    missing_values_count_filtered = filtered_data.isna().sum().sort_values()
    print("Missing values for each column (2014-2024, sorted low to high):")
    print(missing_values_count_filtered)


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
