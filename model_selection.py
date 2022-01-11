"""
Model Selection Module
"""

import importlib

class ModelSelection:
    """
    Model Selection
    """
    def __init__(self):
        self.models = {
            'lenet_modified': {
                'class': 'OurLeNet',
                'canonical': 'Our LeNet'
            },
            'vgg16': {
                'class': 'Vgg16',
                'canonical': 'VGGnet16'
            }
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

    def get_model(self, name: str):
        """
        Build model object
        """
        return getattr(importlib.import_module(f'models.{name}'), self.models[name]['class'])
