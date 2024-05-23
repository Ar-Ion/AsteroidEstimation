from .matcher import Matcher
from .metrics import MatchMetric

class ClassicalMatcher(Matcher):
    def __init__(self, metric, criterion=None):
        super().__init__(criterion=criterion)
        self._metric = MatchMetric.instance(metric)
        
    def match(self, data):
        pred_dists = self._metric.dist(data.prev_features, data.next_features)
        pred_matches = self._criterion.apply(pred_dists)
                
        return pred_dists, pred_matches