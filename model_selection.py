"""
Model Selection Module
"""

import importlib
from abc import ABCMeta

class ModelSelectionInterface(metaclass=ABCMeta):
    """
    ModelSelection Constant Definitions Interface
    """
    # VGG
    VGG_NAME = 'vgg19'
    VGG_CLASS = 'Vgg19'
    VGG_CANONICAL = 'VGG19 (TL)'
    # VGG Cherry
    VGG_CHERRY_NAME = 'vgg19_cherry'
    VGG_CHERRY_CLASS = 'Vgg19Cherry'
    VGG_CHERRY_CANONICAL = 'VGG19 (TL+Cherry-picked)'
    # Inception
    INCEPTION_NAME = 'inceptionv3'
    INCEPTION_CLASS = 'InceptionV3'
    INCEPTION_CANONICAL = 'InceptionV3 (TL)'
    # Inception Cherry
    INCEPTION_CHERRY_NAME = 'inceptionv3_cherry'
    INCEPTION_CHERRY_CLASS = 'InceptionV3Cherry'
    INCEPTION_CHERRY_CANONICAL = 'InceptionV3 (TL+Cherry-picked)'
    # MobileNet
    MOBILENET_NAME = 'mobilenetv2'
    MOBILENET_CLASS = 'MobileNetV2'
    MOBILENET_CANONICAL = 'MobileNetV2 (TL)'
    # MobileNet Cherry
    MOBILENET_CHERRY_NAME = 'mobilenetv2_cherry'
    MOBILENET_CHERRY_CLASS = 'MobileNetV2Cherry'
    MOBILENET_CHERRY_CANONICAL = 'MobileNetV2 (TL+Cherry-picked)'

class ModelSelection(ModelSelectionInterface):
    """
    Model Selection
    """
    def __init__(self):
        self.models = {
            self.VGG_NAME: {
                'class': self.VGG_CLASS,
                'canonical': self.VGG_CANONICAL
            },
            self.VGG_CHERRY_NAME: {
                'class': self.VGG_CHERRY_CLASS,
                'canonical': self.VGG_CHERRY_CANONICAL
            },
            self.INCEPTION_NAME: {
                'class': self.INCEPTION_CLASS,
                'canonical': self.INCEPTION_CANONICAL
            },
            self.INCEPTION_CHERRY_NAME: {
                'class': self.INCEPTION_CHERRY_CLASS,
                'canonical': self.INCEPTION_CHERRY_CANONICAL
            },
            self.MOBILENET_NAME: {
                'class': self.MOBILENET_CLASS,
                'canonical': self.MOBILENET_CANONICAL
            },
            self.MOBILENET_CHERRY_NAME: {
                'class': self.MOBILENET_CHERRY_CLASS,
                'canonical': self.MOBILENET_CHERRY_CANONICAL
            },
        }

    def get_model_keys(self):
        """
        Obtain Available Model Keys
        """
        return list(self.models.keys())

    def get_model_canonical_name(self, name):
        """
        Obtain the regular name for the model
        """
        return self.models[name]['canonical']

    def get_models(self):
        """
        Obtain Complete Model Dict
        """
        return self.models

    def get_profiles(self):
        """
        Get Full Selection Predefined Profile
        """
        return [
            [self.VGG_NAME, self.VGG_CHERRY_NAME],
            [self.INCEPTION_NAME, self.INCEPTION_CHERRY_NAME],
            [self.MOBILENET_NAME, self.MOBILENET_CHERRY_NAME],
        ]

    def get_model(self, name: str):
        """
        Build model object
        """
        return getattr(importlib.import_module(f'models.{name}'), self.models[name]['class'])
