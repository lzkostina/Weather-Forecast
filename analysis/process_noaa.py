import pandas as pd
import os
import pandas as pd

# Define the column names and their respective column positions
columns = ['ID', 'YEAR', 'MONTH', 'ELEMENT'] + [f'VALUE{i}' for i in range(1, 32)] + [f'MFLAG{i}' for i in range(1, 32)] + [f'QFLAG{i}' for i in range(1, 32)] + [f'SFLAG{i}' for i in range(1, 32)]

# Define the fixed column positions for each variable
column_positions = [
    (0, 11),  # ID (1-11)
    (11, 15),  # YEAR (12-15)
    (15, 17),  # MONTH (16-17)
    (17, 21)  # ELEMENT (18-21)
] + [(i*5+21, i*5+26) for i in range(31)] * 3  # VALUE1 to VALUE31, MFLAG1 to MFLAG31, QFLAG1 to QFLAG31, SFLAG1 to SFLAG31


# Function to parse a single line of the file
def parse_line(line):
    data = {}
    
    # Extract the values for each column based on their positions
    data['ID'] = line[0:11].strip()
    data['YEAR'] = int(line[11:15].strip())
    data['MONTH'] = int(line[15:17].strip())
    data['ELEMENT'] = line[17:21].strip()

    # Extract VALUE, MFLAG, QFLAG, SFLAG columns
    for i in range(31):
        value_str = line[21 + i*5:26 + i*5].strip()
        
        # Try to convert the value to an integer, but handle non-numeric values
        try:
            data[f'VALUE{i+1}'] = int(value_str) if value_str else None  # Assign None if the value is empty
        except ValueError:
            data[f'VALUE{i+1}'] = None  # If there's a conversion error, assign None
        
        # Extract MFLAG, QFLAG, SFLAG (which should be single characters)
        data[f'MFLAG{i+1}'] = line[26 + i*5:27 + i*5].strip()
        data[f'QFLAG{i+1}'] = line[27 + i*5:28 + i*5].strip()
        data[f'SFLAG{i+1}'] = line[28 + i*5:29 + i*5].strip()
    
    return data


def convert_dly_to_dataframe(input_dir, output_dir, parse_line, file_extension="csv"):
    """
    Converts all .dly files from the input directory to DataFrames and saves them in the output directory.
    
    Parameters:
        input_dir (str): Path to the directory containing .dly files.
        output_dir (str): Path to the directory where DataFrames will be saved.
        parse_line (function): Function that parses a line in the .dly file and returns a dictionary.
        file_extension (str): Format to save the DataFrame (e.g., 'csv' or 'parquet'). Defaults to 'csv'.
    """
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    for filename in os.listdir(input_dir):
        if filename.endswith(".dly"):
            file_path = os.path.join(input_dir, filename)
            
            # Read and parse the file into a list of dictionaries
            records = []
            with open(file_path, 'r') as f:
                for line in f:
                    record = parse_line(line)
                    records.append(record)
            
            # Convert to DataFrame
            df = pd.DataFrame(records)
            
            # Save the DataFrame
            output_file_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.{file_extension}")
            if file_extension == "csv":
                df.to_csv(output_file_path, index=False)
            elif file_extension == "parquet":
                df.to_parquet(output_file_path, index=False)
            
            print(f"Saved {output_file_path}")


input_dir = 'data/original'
output_dir = 'data/processed'

convert_dly_to_dataframe(input_dir, output_dir, parse_line)

