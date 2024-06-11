import torch
from .metric import MatchMetric

class Hamming(MatchMetric):
    def dist(a, b):
        return torch.cdist(a.float(), b.float(), p=0).to(dtype=torch.float32)