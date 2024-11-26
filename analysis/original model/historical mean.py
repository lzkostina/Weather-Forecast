import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_predictions(actual, predicted, num_features=1):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)  # R² calculation
    n = len(actual)  # Number of data points
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - num_features - 1)  # Adjusted R²
    return mse, mae, r2, adj_r2

# Define Historical Mean prediction function
def historical_mean_forecast(series, steps=5, start_date=None):
    mean_value = series.mean()
    forecast_values = [mean_value] * steps
    future_dates = [start_date + pd.Timedelta(days=i) for i in range(1, steps + 1)]
    forecast_df = pd.DataFrame({'DATE': future_dates, 'FORECAST': forecast_values})
    return forecast_df


directory = '.'  
output_file = 'output/AllCities_HistoricalMean.txt'  
os.makedirs('output', exist_ok=True)  

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("Historical Mean Results for All Cities:\n\n")

for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        city_name = os.path.splitext(filename)[0]
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)

        df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']], errors='coerce')
        # just for test
        # df = df[df['DATE'] <= '2024-11-19']  # Filter data before or on 2024-11-19
        df['TAVG'] = (df['TMAX'] + df['TMIN']) / 2  
        df['TMAX'] = df['TMAX'].interpolate(method='linear')
        df['TMIN'] = df['TMIN'].interpolate(method='linear')
        df['TAVG'] = df['TAVG'].interpolate(method='linear')

        # Train-Test Split
        train_tmax = df['TMAX'][:-5]
        test_tmax = df['TMAX'][-5:]
        train_tmin = df['TMIN'][:-5]
        test_tmin = df['TMIN'][-5:]
        train_tavg = df['TAVG'][:-5]
        test_tavg = df['TAVG'][-5:]

        start_date = df['DATE'].max()  # Start date for predictions
        tmax_historical_forecast_df = historical_mean_forecast(train_tmax, steps=5, start_date=start_date)
        tmin_historical_forecast_df = historical_mean_forecast(train_tmin, steps=5, start_date=start_date)
        tavg_historical_forecast_df = historical_mean_forecast(train_tavg, steps=5, start_date=start_date)

        tmax_mse, tmax_mae, tmax_r2, tmax_adj_r2 = evaluate_predictions(test_tmax, tmax_historical_forecast_df['FORECAST'])
        tmin_mse, tmin_mae, tmin_r2, tmin_adj_r2 = evaluate_predictions(test_tmin, tmin_historical_forecast_df['FORECAST'])
        tavg_mse, tavg_mae, tavg_r2, tavg_adj_r2 = evaluate_predictions(test_tavg, tavg_historical_forecast_df['FORECAST'])

        combined_forecast_df = pd.DataFrame({
            'DATE': tmax_historical_forecast_df['DATE'],
            'TMIN': tmin_historical_forecast_df['FORECAST'],
            'TMAX': tmax_historical_forecast_df['FORECAST'],
            'TAVG': tavg_historical_forecast_df['FORECAST']
        })

        output_text = f"City: {city_name}\n"
        output_text += f"TMAX - MSE: {tmax_mse:.2f}, MAE: {tmax_mae:.2f}, R²: {tmax_r2:.4f}, Adjusted R²: {tmax_adj_r2:.4f}\n"
        output_text += f"TMIN - MSE: {tmin_mse:.2f}, MAE: {tmin_mae:.2f}, R²: {tmin_r2:.4f}, Adjusted R²: {tmin_adj_r2:.4f}\n"
        output_text += f"TAVG - MSE: {tavg_mse:.2f}, MAE: {tavg_mae:.2f}, R²: {tavg_r2:.4f}, Adjusted R²: {tavg_adj_r2:.4f}\n\n"
        output_text += "Future 5-Day Predictions (Combined):\n"
        output_text += combined_forecast_df.to_string(index=False)
        output_text += "\n\n" + "-" * 50 + "\n\n"

        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(output_text)

        print(f"Processed {city_name}, results appended to {output_file}")
