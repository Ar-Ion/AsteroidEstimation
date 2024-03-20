from abc import ABC, abstractmethod

class Backend(ABC):    
    @abstractmethod
    def extract_features(self, image):
        pass