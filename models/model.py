"""
Base Prediction Models
"""

from typing import Optional
from enum import Enum
from abc import ABC, abstractmethod
from cv2 import cv2
import numpy as np

class Prediction(Enum):
    """
    Face Prediction Enumeration Data
    """
    DROWSY = 0
    ALERT  = 1

class Model(ABC):
    """
    Base Model Class for Face Detection Models
    """
    FACE_SHAPE = (224, 224)

    def __init__(self, face: Optional[np.ndarray] = None) -> None:
        self.model = self._load_model()
        self.face = None
        self._set_face(face)

    def _set_face(self, face: Optional[np.ndarray]):
        """
        Set Face Image
        """
        if face is None:
            return self
        self.face = face
        return self

    def _resize_image(self):
        self.face = cv2.resize(self.face, self.FACE_SHAPE) / 255

    def _preprocess(self):
        """
        Preprocess face data
        """
        if self.face is None:
            return self
        self._resize_image()
        return self

    def _get_confidence(self, raw_classification_value: int): # pylint: disable=no-self-use
        """
        Get confidence value in percentage
        """
        return np.abs(raw_classification_value - .5) * 2

    def _get_prediction_result(self, raw_classification_value: int):
        """
        Get prediction result
        """
        confidence = self._get_confidence(raw_classification_value)
        if int(np.round(raw_classification_value, 0)):
            return {
                'prediction': Prediction.DROWSY,
                'confidence': confidence
            }
        return {
            'prediction': Prediction.ALERT,
            'confidence': confidence
        }

    def trigger_predict(self, face: Optional[np.ndarray]):
        """
        Public Trigger Predict Method
        """
        self._set_face(face)._preprocess() # pylint: disable=W0212
        if self.face is not None:
            return self.predict()
        return None

    @abstractmethod
    def _load_model(self):
        """
        Load the model
        """

    @abstractmethod
    def predict(self) -> Prediction:
        """
        Predict the face
        """
