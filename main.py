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

    model = TestPredictor()
    data = []
    predictions = model.predict(data)
    
    prediction_date = f"{datetime.date.today():%Y-%m-%d}"
    predictions_rounded = np.around(predictions, 1)
    
    fmt_str_contents = [prediction_date] + list([str(prediction) for prediction in predictions_rounded])
    fmt_str = ", ".join(fmt_str_contents)
    print(fmt_str)