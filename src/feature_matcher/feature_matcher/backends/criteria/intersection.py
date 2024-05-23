from .criterion import MatchCriterion

class Intersection(MatchCriterion):
    def __init__(self, crit1, crit2):
        self._crit1 = crit1
        self._crit2 = crit2
        
    def apply(self, dist_matrix):
        return self._crit1.apply(dist_matrix) * self._crit2.apply(dist_matrix)