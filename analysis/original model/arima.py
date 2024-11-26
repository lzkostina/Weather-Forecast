import pandas as pd
import numpy as np
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

def arima_forecast(series, order=(1, 1, 1), steps=5, start_date=None):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    future_dates = [start_date + pd.Timedelta(days=i) for i in range(1, steps + 1)]
    forecast_df = pd.DataFrame({'DATE': future_dates, 'FORECAST': forecast})
    return forecast_df

def evaluate_predictions(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    return mse, mae

directory = '.'  
output_file = 'output/AllCities_ARIMA.txt' 
os.makedirs('output', exist_ok=True)  

def combine_future_predictions(tmin_forecast_df, tmax_forecast_df, tavg_forecast_df):
    """Combine TMIN, TMAX, and TAVG forecasts into a single DataFrame."""
    combined_forecast = pd.DataFrame({
        'DATE': tmin_forecast_df['DATE'],
        'TMIN': tmin_forecast_df['FORECAST'],
        'TMAX': tmax_forecast_df['FORECAST'],
        'TAVG': tavg_forecast_df['FORECAST']
    })
    return combined_forecast

for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        city_name = os.path.splitext(filename)[0]
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)

        df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']], errors='coerce')

        # need to be corrected
        # just for test
        df = df[df['DATE'] <= '2024-11-19']  # Filter data before or on 2024-11-19

        df['TMAX'] = df['TMAX'].replace(0, np.nan)
        df['TMIN'] = df['TMIN'].replace(0, np.nan)
        df['TAVG'] = (df['TMAX'] + df['TMIN']) / 2  
        df['TMAX'] = df['TMAX'].interpolate(method='linear')
        df['TMIN'] = df['TMIN'].interpolate(method='linear')
        df['TAVG'] = df['TAVG'].interpolate(method='linear')

        # Train-Test Split
        train_tmax = df['TMAX'][-600:-5]
        test_tmax = df['TMAX'][-5:]
        train_tmin = df['TMIN'][-600:-5]
        test_tmin = df['TMIN'][-5:]
        train_tavg = df['TAVG'][-600:-5]
        test_tavg = df['TAVG'][-5:]

        start_date = df['DATE'].max()
        tmax_arima_forecast_df = arima_forecast(train_tmax, order=(5, 1, 3), steps=5, start_date=start_date)
        tmin_arima_forecast_df = arima_forecast(train_tmin, order=(5, 1, 1), steps=5, start_date=start_date)
        tavg_arima_forecast_df = arima_forecast(train_tavg, order=(5, 1, 3), steps=5, start_date=start_date)

        combined_forecast_df = combine_future_predictions(tmin_arima_forecast_df, tmax_arima_forecast_df, tavg_arima_forecast_df)

        # Evaluate ARIMA predictions
        tmax_arima_mse, tmax_arima_mae = evaluate_predictions(test_tmax, tmax_arima_forecast_df['FORECAST'])
        tmin_arima_mse, tmin_arima_mae = evaluate_predictions(test_tmin, tmin_arima_forecast_df['FORECAST'])
        tavg_arima_mse, tavg_arima_mae = evaluate_predictions(test_tavg, tavg_arima_forecast_df['FORECAST'])

        output_text = f"City: {city_name}\n"
        output_text += f"TMAX - MSE: {tmax_arima_mse:.2f}, MAE: {tmax_arima_mae:.2f}\n"
        output_text += f"TMIN - MSE: {tmin_arima_mse:.2f}, MAE: {tmin_arima_mae:.2f}\n"
        output_text += f"TAVG - MSE: {tavg_arima_mse:.2f}, MAE: {tavg_arima_mae:.2f}\n\n"
        output_text += "Future 5-Day Predictions (Combined):\n"
        output_text += combined_forecast_df.to_string(index=False)
        output_text += "\n\n" + "-" * 50 + "\n\n"
        
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(output_text)

        print(f"Processed {city_name}, results appended to {output_file}")

