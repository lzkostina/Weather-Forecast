import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


def evaluate_model(y_test, y_pred, X_test):
    n = len(y_test)
    p = X_test.shape[1]
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return r2, adj_r2, mse, rmse

def generate_future_predictions(data, model_min, model_max, model_avg, days=5, start_date=None, lag_features=[]):
    future_predictions = []
    current_data = data.iloc[-1][lag_features].values.reshape(1, -1)
    current_date = pd.to_datetime(start_date)

    for day in range(1, days + 1):
        pred_min = model_min.predict(current_data)[0]
        pred_max = model_max.predict(current_data)[0]
        pred_avg = model_avg.predict(current_data)[0]

        future_predictions.append({
            'DATE': current_date + pd.Timedelta(days=day),
            'TMIN': pred_min,
            'TMAX': pred_max,
            'TAVG': pred_avg
        })

        current_data = np.roll(current_data, -3)  
        current_data[0, :3] = [pred_max, pred_min, pred_avg]  # Insert new predictions

    return pd.DataFrame(future_predictions)

directory = '.'  
output_file = 'output/AllCities_XGBoost.txt' 
os.makedirs('output', exist_ok=True)  

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("XGBoost Model Results for All Cities:\n\n")

for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        city_name = os.path.splitext(filename)[0]
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)
        df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']], errors='coerce')
        # Filter data before or on 2024-11-19
        # need to be corrected
        # just for test
        # df = df[df['DATE'] <= '2024-11-19']

        df['TAVG'] = (df['TMAX'] + df['TMIN']) / 2
        for i in range(1, 6):
            df[f'TMAX_lag{i}'] = df['TMAX'].shift(i)
            df[f'TMIN_lag{i}'] = df['TMIN'].shift(i)
            df[f'TAVG_lag{i}'] = df['TAVG'].shift(i)

        df.dropna(subset=['TMAX', 'TMIN', 'TAVG'], inplace=True)

        lag_features = [f'TMAX_lag{i}' for i in range(1, 6)] + \
                       [f'TMIN_lag{i}' for i in range(1, 6)] + \
                       [f'TAVG_lag{i}' for i in range(1, 6)] + ['PRCP', 'SNOW', 'SNWD']

        X = df[lag_features]
        y_min = df['TMIN']
        y_max = df['TMAX']
        y_avg = df['TAVG']

        # Train-Test Split
        # test_size=0.2
        X_train, X_test, y_min_train, y_min_test, y_max_train, y_max_test, y_avg_train, y_avg_test = train_test_split(
            X, y_min, y_max, y_avg, test_size=0.2, random_state=42
        )
        
        model_min = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42)
        model_max = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42)
        model_avg = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42)

        model_min.fit(X_train, y_min_train)
        model_max.fit(X_train, y_max_train)
        model_avg.fit(X_train, y_avg_train)

        y_min_pred = model_min.predict(X_test)
        y_max_pred = model_max.predict(X_test)
        y_avg_pred = model_avg.predict(X_test)

        r2_min, adj_r2_min, mse_min, rmse_min = evaluate_model(y_min_test, y_min_pred, X_test)
        r2_max, adj_r2_max, mse_max, rmse_max = evaluate_model(y_max_test, y_max_pred, X_test)
        r2_avg, adj_r2_avg, mse_avg, rmse_avg = evaluate_model(y_avg_test, y_avg_pred, X_test)

        future_predictions = generate_future_predictions(df, model_min, model_max, model_avg, days=5, start_date='2024-11-19', lag_features=lag_features)

        output_text = f"City: {city_name}\n"
        output_text += f"TMIN Evaluation: R²: {r2_min:.4f}, Adjusted R²: {adj_r2_min:.4f}, MSE: {mse_min:.4f}, RMSE: {rmse_min:.4f}\n"
        output_text += f"TMAX Evaluation: R²: {r2_max:.4f}, Adjusted R²: {adj_r2_max:.4f}, MSE: {mse_max:.4f}, RMSE: {rmse_max:.4f}\n"
        output_text += f"TAVG Evaluation: R²: {r2_avg:.4f}, Adjusted R²: {adj_r2_avg:.4f}, MSE: {mse_avg:.4f}, RMSE: {rmse_avg:.4f}\n\n"
        output_text += "Future 5-Day Predictions:\n"
        output_text += future_predictions.to_string(index=False)
        output_text += "\n\n" + "-" * 50 + "\n\n"


        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(output_text)


        print(f"Processed {city_name}, results appended to {output_file}")
