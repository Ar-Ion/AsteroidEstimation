from abc import ABC, abstractmethod
import importlib

class Matcher(ABC):
    
    def __init__(self, criterion=None):
        self._criterion = criterion
    
    @abstractmethod
    def match(self, data):
        pass
    
    def set_criterion(self, crit):
        self._criterion = crit
        
    def instance(type, *args):
        module = importlib.import_module("feature_matcher.backends")
        matcher_class = getattr(module, type)
        return matcher_class(*args)