import torch
from .metric import MatchMetric

class Cosine(MatchMetric):
    def dist(a, b):
        return (a @ b.T).to(dtype=torch.float32)