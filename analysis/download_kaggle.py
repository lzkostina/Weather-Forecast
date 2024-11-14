import sys
import os
import subprocess


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

if __name__ == "__main__":

    # Define the directory and create it if it doesn't exist
    save_directory = "data/original"
    os.makedirs(save_directory, exist_ok=True)
    print(f"Directory '{save_directory}' created or already exists.")


    # Run the Kaggle CLI command to download the dataset
    result = subprocess.run([
        "kaggle", "datasets", "download", 
        "-d", "sudalairajkumar/daily-temperature-of-major-cities", 
        "-p", save_directory
    ])

    if result.returncode == 0:
        print("Dataset downloaded successfully!")
    else:
        print("Failed to download the dataset.")
