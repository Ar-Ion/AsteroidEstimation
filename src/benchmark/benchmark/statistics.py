from abc import ABC, abstractmethod
import torch

class MatchMetric(ABC):
    @abstractmethod
    def dist(a, b):
        pass

class L2(MatchMetric):
    def dist(a, b):
        return torch.cdist(a.float(), b.float()).to(dtype=torch.float16)
    
class Cosine(MatchMetric):
    def dist(a, b):
        return (a @ b.T).to(dtype=torch.float16)
    
class MatchCriterion(ABC):
    @abstractmethod
    def apply(self, dist_matrix):
        pass
    
class LowerThanCriterion(MatchCriterion):
    def __init__(self, epsilon):
        self._epsilon = epsilon
    
    def apply(self, dist_matrix):
        match_matrix = (dist_matrix < self._epsilon)
        return match_matrix.to(dtype=torch.float16)

class GreaterThanCriterion(MatchCriterion):
    def __init__(self, epsilon):
        self._epsilon = epsilon
        
    def apply(self, dist_matrix):
        match_matrix = (dist_matrix > self._epsilon)
        return match_matrix.to(dtype=torch.float16)

class RatioCriterion(MatchCriterion):
    def __init__(self, ratio):
        self._ratio = ratio
    
    def apply(self, dist_matrix):
        top = torch.topk(dist_matrix, 2, dim=0)
        match_matrix = dist_matrix > self._ratio*top.values[1]
        return match_matrix.to(dtype=torch.float16)
    
class PassThroughCriterion(MatchCriterion):
    def apply(self, dist_matrix):
        return dist_matrix
    
    
class Matcher:
    def __init__(self, metric, criterion):
        self._metric = metric
        self._criterion = criterion
        
    def match(self, a, b):
        dist_matrix = self._metric.dist(a, b)
        match_matrix = self._criterion.apply(dist_matrix)
        return dist_matrix, match_matrix

class Statistics:
    def __init__(self, true_matches, pred_matches):
        self._tp = (true_matches * pred_matches).mean()
        self._tn = ((1-true_matches) * (1-pred_matches)).mean()
        self._fp = ((1-true_matches) * pred_matches).mean()
        self._fn = (true_matches * (1-pred_matches)).mean()
        
    def true_positives(self):
        return self._tp

    def true_negatives(self):
        return self._tn
    
    def false_positives(self):
        return self._fp
    
    def false_negatives(self):
        return self._fn
    
    def accuracy(self):
        return self._tp + self._tn
    
    def precision(self):
        return self._tp / (self._tp + self._fp)

    def recall(self):
        return self._tp / (self._tp + self._fn)
    
    def f1(self):
        precision = self.precision()
        recall = self.recall()
        return 2*(precision*recall)/(precision+recall)