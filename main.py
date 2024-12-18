import os
import sys

import predictor.test_predictor

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import datetime

from predictor import utils
from predictor import test_predictor
from predictor.make_predictions import make_predictions_all_stations

import logging


if __name__ == "__main__":

    # model = test_predictor.TestPredictor()
    model = predictor.test_predictor.RidgeRegressionPredictor("predictor/models/RidgeRegression/model_full")

    # get current year, month, and day
    current_date = datetime.date.today()
    year = current_date.year
    month = current_date.month
    day = current_date.day

    # make predictions for all stations
    try:
        predictions = make_predictions_all_stations(model, year, month, day)
        #predictions = model.predict(None, None)
        
    except Exception as e:
        logging.error(f"Error making predictions: {e}")
        sys.exit(1)
    
    # Ensure the predictions are numeric
    predictions_rounded = np.around(predictions, 1)
    
    prediction_date = f"{datetime.date.today():%Y-%m-%d}"
    
    fmt_str_contents = [prediction_date] + list([str(prediction) for prediction in predictions_rounded])
    fmt_str = ", ".join(fmt_str_contents)
    print(fmt_str)



#
# import os
# import sys
#
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
#
# import numpy as np
# import datetime
#
# from predictor import utils
# from predictor import test_predictor
# from predictor.test_predictor import Predictor
# from predictor.test_predictor import TestPredictor
#
# import logging
#
#
# if __name__ == "__main__":
#
#     model = TestPredictor()
#
#     # Example input for `data` and `station` (not used in this TestPredictor but required for method signature)
#     data = []  # Empty or mock data
#     station = None  # Placeholder for station
#
#     # Call predict with both arguments
#     predictions = model.predict(data, station)
#
#     # Ensure the predictions are numeric (already zeros, so this is optional)
#     predictions_rounded = np.around(predictions, 1)
#
#     # Get today's date
#     prediction_date = f"{datetime.date.today():%Y-%m-%d}"
#
#     # Format the output
#     fmt_str_contents = [prediction_date] + list([str(prediction) for prediction in predictions_rounded])
#     fmt_str = ", ".join(fmt_str_contents)
#
#     # Print the formatted output
#     print(fmt_str)
