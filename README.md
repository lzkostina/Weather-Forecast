# Weather-Forecast

This repository contains the code, data, and documentation for our weather forecasting project, which uses historical and current meteorological data to predict weather conditions over a five-day horizon.  

### Key Features  
- **Data Sources**:  
  - NOAA: Historical weather data from 20 stations (1948–2024) processed from `.dly` to tabular format.  
  - OpenWeather: Hourly historical weather data (1979–2024) retrieved via API and aggregated into daily summaries.  

- **Data Preprocessing**:  
  - NOAA: Cleaned and converted data, handling missing values and outliers, and converted temperatures to Fahrenheit.  
  - OpenWeather: Aggregated hourly data to daily format, calculated key features, and merged with NOAA data for training.  

- **Modeling and Analysis**:  
  - Treated the problem as a regression task, predicting future weather conditions (TMIN, TAVG, TMAX) based on past data.  
  - Implemented and evaluated several predictors:  
    - Baseline Models: Previous Day Predictor, Historical Average Predictor.  
    - Regression Models: Linear Regression, Ridge Regression, LASSO.  
    - Ensemble Models: Random Forest, Weighted Average Predictor.  
  - Final Model: Ridge Regression, optimized for minimal Mean Squared Error (MSE).  

- **Results**:  
  - Strong performance for short-term predictions (days 1 and 2).  
  - Identified challenges in maintaining accuracy for longer horizons (days 3–5).  

- **Reproducibility**:  
  - All scripts for data retrieval, preprocessing, and model training can be found in the `analysis` folder.  
  - Raw and processed datasets are stored in the `data` folder.  
  - Docker image included for seamless setup and reproducibility.  

 - **Command line**

  - git pull -to copy everything from git repo 

  - make clean_data -to clean data folder

  - make clean -- deletes everything except for the code (i.e., markdown files) and raw data (as originally downloaded)

  - make download_data -to download kaggle and noaa data in their original format

  - make process_data -to convert original data into csv formats and put these files in data/processed folder

  - docker build -t weather-forecast . -to create a docker image 

  - docker run --it --rm weather-forecast make predictions  -to display predictions
    
  - docker run --it --rm weather-forecast make predictions  >> predictions.csv  -to save predictions into predictions.csv file 
