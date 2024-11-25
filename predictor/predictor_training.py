# Training predictors from test_predictor.py

import numpy as np
import pandas as pd
import os
import sys
import datetime
import joblib
from utils import stations_list
from test_predictor import TestPredictor
from test_predictor import PreviousDayPredictor
from test_predictor import AverageLastWeekPredictor
from test_predictor import Predictor
from test_predictor import LinearRegressionPredictor



