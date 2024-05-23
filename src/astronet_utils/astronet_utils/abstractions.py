from abc import ABC, abstractmethod, staticmethod

class Batchable(ABC):
    @abstractmethod
    @staticmethod
    def retrieve(data, index):
        pass
    
    @abstractmethod
    @staticmethod
    def batched(lst):
        pass
    
class Torchable(ABC):
    @abstractmethod
    @staticmethod
    def to(data, device=None, dtype=None):
        pass
    
    @abstractmethod
    @staticmethod
    def detach(data):
        pass
    
class Chunkable(ABC):
    @abstractmethod
    @staticmethod
    def create_chunks(data, in_dim, out_dim):
        pass