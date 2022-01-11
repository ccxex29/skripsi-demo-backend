"""
ModelMemo Module
"""

class ModelMemo:
    """
    Model Memoisation
    """
    def __init__(self):
        self.__model = {}

    def add_memo(self, name: str, model):
        """
        Add memo helper method
        """
        if name not in self.__model:
            self.__model[name] = model

    def get_memo(self, name: str):
        """
        Memo setter method
        """
        return self.__model[name]

    def has_memo(self, name: str):
        """
        Returns whether memo has been assigned
        """
        return name in self.__model
