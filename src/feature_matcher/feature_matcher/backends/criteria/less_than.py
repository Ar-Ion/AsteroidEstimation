import torch
from .criterion import MatchCriterion

class LessThan(MatchCriterion):
    def __init__(self, epsilon):
        self._epsilon = epsilon
    
    def apply(self, dist_matrix):
        match_matrix = (dist_matrix < self._epsilon)
        return match_matrix.to(dtype=torch.float32)