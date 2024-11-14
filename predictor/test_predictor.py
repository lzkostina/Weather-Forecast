import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))


import numpy as np

from abc import ABC, abstractmethod
 
class Predictor(ABC): 
    @abstractmethod
    def predict(self, data):
        """
        """
        pass

class TestPredictor(Predictor):
    def __init__(self):
        pass

    def predict(self, data):
        return np.zeros(300)