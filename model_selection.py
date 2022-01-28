"""
Model Selection Module
"""

import importlib
from abc import ABCMeta

class ModelSelectionInterface(metaclass=ABCMeta):
    """
    ModelSelection Constant Definitions Interface
    """
    VGG_NAME = 'vgg16'
    VGG_CLASS = 'Vgg16'
    VGG_CANONICAL = 'VGG16 (TL)'
    VGG_CHERRY_NAME = 'vgg16_cherry'
    VGG_CHERRY_NAME_CLASS = 'Vgg16Cherry'
    VGG_CHERRY_NAME_CANONICAL = 'VGG16 (TL+Cherry-picked)'

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
            #self.VGG_CHERRY_NAME: {
            #    'class': self.VGG_CLASS,
            #    'canonical': self.VGG_CHERRY_NAME_CANONICAL
            #},
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
            [self.VGG_NAME, self.VGG_NAME]
        ]

    def get_model(self, name: str):
        """
        Build model object
        """
        return getattr(importlib.import_module(f'models.{name}'), self.models[name]['class'])
