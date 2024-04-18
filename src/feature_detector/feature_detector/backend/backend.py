from abc import ABC, abstractmethod

class Backend(ABC):    
    @abstractmethod
    def detect_features(self, image):
        pass