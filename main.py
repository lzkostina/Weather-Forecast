# import os
# import sys

# import predictor.test_predictor

# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# import numpy as np
# import datetime

# from predictor import utils
# from predictor import test_predictor
# from predictor.make_predictions import make_predictions_all_stations

# import logging


# if __name__ == "__main__":

#     model = test_predictor.TestPredictor()
#     #model = predictor.test_predictor.LinearRegressionPredictor()

#     # get current year, month, and day
#     current_date = datetime.date.today()
#     year = current_date.year
#     month = current_date.month
#     day = current_date.day

#     # make predictions for all stations
#     try:
#         #predictions = make_predictions_all_stations(model, year, month, day)
#         predictions = model.predict(None, None)
        
#     except Exception as e:
#         logging.error(f"Error making predictions: {e}")
#         sys.exit(1)
    
#     # Ensure the predictions are numeric
#     predictions_rounded = np.around(predictions, 1)
    
#     prediction_date = f"{datetime.date.today():%Y-%m-%d}"
    
#     fmt_str_contents = [prediction_date] + list([str(prediction) for prediction in predictions_rounded])
#     fmt_str = ", ".join(fmt_str_contents)
#     print(fmt_str)




import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import datetime

from predictor import utils
from predictor import test_predictor
from predictor.test_predictor import Predictor
from predictor.test_predictor import TestPredictor

import logging


if __name__ == "__main__":

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    model = TestPredictor()
=======
    model = predictor.TestPredictor()
>>>>>>> 41a27f6ba1f26618a5c1d4d49b0ffe3ee6c5fb91
=======
    model = test_predictor.TestPredictor()
>>>>>>> 1c3621c7a5fca23c3af196ff52e777b8a3f8cf79
    #model = predictor.test_predictor.LinearRegressionPredictor()

    # get current year, month, and day
    current_date = datetime.date.today()
    year = current_date.year
    month = current_date.month
    day = current_date.day

    # make predictions for all stations
    try:
        #predictions = make_predictions_all_stations(model, year, month, day)
        predictions = model.predict(None, None)
        
    except Exception as e:
        logging.error(f"Error making predictions: {e}")
        sys.exit(1)
=======
    model = TestPredictor()
<<<<<<< HEAD
    data = []
<<<<<<< HEAD
    predictions = model.predict(data)
>>>>>>> c7bd8e9d9d4da17e0bf8dbd5229c0011dd40606b
=======
    station = None
    predictions = model.predict(data, station)
>>>>>>> 44fcec22fda7ed8c35aabc254c6efa6789e48ded
    
=======
        
    data = [] 
    station = None  

    predictions = model.predict(data, station)
>>>>>>> 5633b4665d8c8272a8ed94952084092ffc430c8a
    predictions_rounded = np.around(predictions, 1)

    prediction_date = f"{datetime.date.today():%Y-%m-%d}"

    fmt_str_contents = [prediction_date] + list([str(prediction) for prediction in predictions_rounded])
    fmt_str = ", ".join(fmt_str_contents)

    print(fmt_str)
