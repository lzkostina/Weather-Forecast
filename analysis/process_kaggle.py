import pandas as pd
import zipfile
import os

# Path to the .csv.zip file
file_path = "data/original/daily-temperature-of-major-cities.zip" 
# Open the ZIP file
with zipfile.ZipFile(file_path, 'r') as z:
    # List all files in the ZIP to find the correct .csv file
    print("Files in ZIP:", z.namelist())
    
    # Extract the specific .csv file and read it into a DataFrame
    with z.open('city_temperature.csv') as f:  # Adjust to your specific file name if different
        df = pd.read_csv(f)

# Display the first few rows of the DataFrame
#print(df.head())

df_0 = df[df['Country'] == 'US'] 

# Path to the folder in your Git repository
folder_path = "data/processed"  
# Ensure the folder exists (if not, create it)
os.makedirs(folder_path, exist_ok=True)

# Path to save the CSV file
file_name = "kaggle_preprocessed.csv"  # Replace with your desired file name
file_path = os.path.join(folder_path, file_name)

# Save the DataFrame to the folder
df_0.to_csv(file_path, index=False)
