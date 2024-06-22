from abc import ABC, abstractmethod

class Batchable(ABC):
    @abstractmethod
    def retrieve(data, index):
        pass
    
    @abstractmethod
    def batched(lst):
        pass
    
class Torchable(ABC):
    @abstractmethod
    def to(data, device=None, dtype=None):
        pass
    
    @abstractmethod
    def detach(data):
        pass
    
class Chunkable(ABC):
    @abstractmethod
    def create_chunks(data, in_dim, out_dim):
        pass
    
class Filterable(ABC):
    @abstractmethod
    def filter(data, filter):
        pass
    
class Transformable(ABC):
    @abstractmethod
    def transform(data, func):
        pass