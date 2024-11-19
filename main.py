import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import datetime

from predictor import utils
from predictor import test_predictor
from predictor.test_predictor import Predictor
from predictor.test_predictor import TestPredictor
from predictor.test_predictor import PreviousDayPredictor

import logging


if __name__ == "__main__":

    #model = TestPredictor()
    model = PreviousDayPredictor()
    data = []
    predictions = model.predict("2023-11-19")
    
    # Ensure the predictions are numeric
    predictions_rounded = np.around(predictions, 1)
    
    prediction_date = f"{datetime.date.today():%Y-%m-%d}"
    
    fmt_str_contents = [prediction_date] + list([str(prediction) for prediction in predictions_rounded])
    fmt_str = ", ".join(fmt_str_contents)
    print(fmt_str)