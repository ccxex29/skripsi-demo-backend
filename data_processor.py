"""
Data Processor Module
"""

from typing import Optional

import logging
import base64
from io import BytesIO
import json

from cerberus import Validator

from cv2 import cv2
import numpy as np
from PIL import Image

from face_detection import get_face
from model_selection import ModelSelection

class DataProcessor:
    """
    Take care of each requested and response data
    """
    NORMAL_MODE = 'normal'
    COMPARE_MODE = 'compare'

    @classmethod
    def get_image_in_base64(cls, image: np.ndarray):
        """
        Get Base64 string from image
        """
        try:
            image_buffer = BytesIO()
            Image.fromarray(image).save(image_buffer, format="JPEG")
            return base64.b64encode(image_buffer.getvalue())
        except AttributeError:
            logging.error('Server Error')
            return None

    def __init__(self, model_selection: Optional[ModelSelection]):
        self.mode = None
        self.architecture_a = None
        self.architecture_b = None
        self.image = None
        self.face_detection = {
            'face_image': None,
            'position': None
        }
        self.validation_result = None
        self.model_selection = model_selection

    def postprocess_message(self, message:str):
        """
        Handle JSON Encoded WS Message to JSON
        """
        validation_schema = {
            'architecture_a': {
                'required': True,
                'type': 'string',
                'allowed': self.model_selection.get_model_keys()
            },
            'architecture_b': {
                'required': True,
                'type': 'string',
                'allowed': self.model_selection.get_model_keys()
            },
            'image': {
                'required': True,
                'type': 'string'
            }
        }
        try:
            parsed_message = json.loads(message)
            self.validation_result = Validator(validation_schema).validate(parsed_message)
            if not self.validation_result:
                logging.debug('Validation Failed')
                return self
            self.architecture_a = parsed_message['architecture_a']
            self.architecture_b = parsed_message['architecture_b']
            self.mode = self.NORMAL_MODE if self.architecture_a == self.architecture_b else self.COMPARE_MODE
            self.image = parsed_message['image']
        except ValueError:
            #self.write_message('Error')
            pass
        except KeyError:
            #self.write_message('Required JSON Key does not exist')
            pass
        return self

    def face_detect(self):
        """
        Postprocess Face Detection
        """
        if not self.validation_result or self.image is None:
            return self
        try:
            face_detection_result = get_face(cv2.imdecode(np.frombuffer(base64.b64decode(self.image), dtype=np.uint8), flags=cv2.IMREAD_COLOR))
            self.face_detection = {
                "face_image": face_detection_result['face'],
                "position": face_detection_result["position"]
            }
        except ValueError:
            logging.warning('[WARN] Invalid Base64')
        except cv2.error:
            logging.warning('[WARN] Invalid Image')
        return self

    def is_normal_mode(self):
        """
        Check if mode is normal
        """
        return self.mode == self.NORMAL_MODE

    def get_face_detection(self):
        """
        Face detection result getter
        """
        return self.face_detection
