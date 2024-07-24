import torch
from .criterion import MatchCriterion

class MinK(MatchCriterion):
    def __init__(self, num):
        self._num = int(num)
        
    def apply(self, dist_matrix):
        flattened = dist_matrix.reshape(1, -1)
        matches = torch.zeros_like(flattened)
        top = torch.topk(flattened, self._num, largest=False)
        matches[:, top.indices] = 1
        return matches.reshape_as(dist_matrix).to(dtype=torch.float32)