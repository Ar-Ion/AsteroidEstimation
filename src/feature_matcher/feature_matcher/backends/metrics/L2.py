import torch
from .metric import MatchMetric

class L2(MatchMetric):
    def dist(a, b):
        return torch.cdist(a.float(), b.float()).to(dtype=torch.float32)