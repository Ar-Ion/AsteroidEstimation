from abc import ABC, abstractmethod

class LossFunction(ABC):
    @abstractmethod
    def loss(self, true_dists, true_matches, pred_dists, pred_matches):
        pass
    
