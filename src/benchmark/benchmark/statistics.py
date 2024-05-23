class Statistics:
    def __init__(self, true_dists, true_matches, pred_matches):
        self._true_count = true_matches.sum()/2 # Note that every match is counted twice, hence the division
        self._positive_count = pred_matches.sum()/2
        self._tp = (true_matches * pred_matches).mean()
        self._tn = ((1-true_matches) * (1-pred_matches)).mean()
        self._fp = ((1-true_matches) * pred_matches).mean()
        self._fn = (true_matches * (1-pred_matches)).mean()
        self._dist = (pred_matches * true_dists).abs().sum() / pred_matches.count_nonzero()
        
    def true_positives(self):
        return self._tp

    def true_negatives(self):
        return self._tn
    
    def false_positives(self):
        return self._fp
    
    def false_negatives(self):
        return self._fn
    
    def true_count(self):
        return self._true_count

    def positive_count(self):
        return self._positive_count
    
    def accuracy(self):
        return self._tp + self._tn
    
    def precision(self):
        return self._tp / (self._tp + self._fp)

    def recall(self):
        return self._tp / (self._tp + self._fn)
    
    def pixel_error(self):
        return self._dist
    
    def f1(self):
        return 2*self._tp / (2*self._tp + self._fp + self._fn)