"""
Face Detector Module
"""

from typing import Union, Optional
from cv2 import cv2 # pylint: disable=E0611
import numpy as np

def get_face(image: np.ndarray) -> dict[str, Optional[Union[np.ndarray, Optional[dict[str, int]]]]]:
    """
    Get Face
    """
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_detector.detectMultiScale(image)
    best_face = {
        "position": {
            "x": None,
            "y": None,
            "w": None,
            "h": None
        },
        "face": None
    }
    for x_pos, y_pos, width, height in faces:
        # Prefer bigger face detection
        if(best_face["face"] is None or height > best_face["face"].shape[0]):
            best_face["position"]["x"] = int(x_pos)
            best_face["position"]["y"] = int(y_pos)
            best_face["position"]["w"] = int(width)
            best_face["position"]["h"] = int(height)
            best_face["face"] = image[y_pos:y_pos+height, x_pos:x_pos+width]
    return best_face
