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
        # Get the current date in "YYYY-MM-DD" format
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Generate 300 numbers of the form "XX.X" (all zeroes with one decimal place)
        predictions = ["{:.1f}".format(num) for num in np.zeros(300)]
        
        # Create the output string
        output = f'"{current_date}" is the current date and then {", ".join(predictions)}'
        
        return output

# Example usage
predictor = TestPredictor()
print(predictor.predict(None))
