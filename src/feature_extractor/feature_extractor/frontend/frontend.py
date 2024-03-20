from abc import ABC, abstractmethod

class Frontend(ABC):
    @abstractmethod
    def loop(self):
        pass
    
    @abstractmethod
    def cleanup(self):
        pass