import sys
import os
from datetime import datetime
import numpy as np
from abc import ABC, abstractmethod

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

class Predictor(ABC): 
    @abstractmethod
    def predict(self, data):
        """
        Abstract method to be implemented by subclasses to perform predictions.
        """
        pass

class TestPredictor(Predictor):
    def __init__(self):
        pass

    def predict(self, data):

        
        # Generate 300 numbers of the form "XX.X" (all zeroes with one decimal place)
        predictions = np.zeros(300)
        #predictions_rounded = np.around(predictions, 1)
        
        # Create the output string
        #output = f'"{current_date}" is the current date and then {", ".join(predictions_rounded)}'
        
        return predictions

# Example usage
#predictor = TestPredictor()
#print(predictor.predict(None))
