"""
Our LeNet Model Module
"""

import numpy as np
from tensorflow import keras
from models.model import Model # pylint: disable=import-error

class Vgg16(Model):
    """
    OurLeNet Model
    """
    FACE_SHAPE = (224, 224)

    def _load_model(self): # pylint: disable=R0201,C0116
        return keras.models.load_model('models/preprocessed/vgg16')

    def predict(self): # pylint: disable=C0116
        return self._get_prediction_result(self.model.predict(np.array([self.face]))[0][0])
