import torch
from .criterion import MatchCriterion

class MinRatio(MatchCriterion):
    def __init__(self, ratio):
        self._ratio = ratio
    
    def apply(self, dist_matrix):
        top = torch.topk(dist_matrix, 2, dim=0, largest=False)
        match_matrix = dist_matrix < self._ratio*top.values[1]
        return match_matrix.to(dtype=torch.float32)