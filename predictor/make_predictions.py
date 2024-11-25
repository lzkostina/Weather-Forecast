# In this file, we will use a given predictor to make predictions for a given day.


import numpy as np
import os
import datetime
import sys
import pandas as pd
from predictor import test_predictor
from predictor import utils

# Helper function to make predictions for a given day

def make_predictions_station(predictor, year, month, day, station):