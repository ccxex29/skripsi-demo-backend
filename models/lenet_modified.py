"""
Our LeNet Model Module
"""

import numpy as np
from tensorflow import keras
from models.model import Model, Prediction

class OurLeNet(Model):
    """
    OurLeNet Model
    """
    def _load_model(self):
        return keras.models.load_model('models/preprocessed/ourlenet')

    def predict(self):
        return self._get_prediction_result(self.model.predict(np.array([self.face]))[0][0])
