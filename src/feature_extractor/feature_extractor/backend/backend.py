from abc import ABC, abstractmethod

class Backend(ABC):    
    @abstractmethod
    def extract_features(self, image):
        pass
    
    @abstractmethod
    def get_match_norm(self):
        pass