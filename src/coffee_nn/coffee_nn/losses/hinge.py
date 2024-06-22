import torch
from .loss import LossFunction

class HingeLoss(LossFunction):
    def __init__(self, margin, weight):
        self._positive_margin = 1.0
        self._negative_margin = margin
        self._weight = weight
        
    def loss(self, true_dists, true_matches, pred_dists, pred_matches):       
        # Hinge loss
                
        fn_loss = true_matches * torch.maximum(torch.tensor(0), self._positive_margin - pred_dists)
        fp_loss = (1 - true_matches) * torch.maximum(torch.tensor(0), pred_dists - self._negative_margin)
                
        maximum_match_count = min(true_dists.shape[0], true_dists.shape[1])
        
        hinge_loss = self._weight*maximum_match_count*fn_loss.sum() + fp_loss.sum()

        return hinge_loss, maximum_match_count
    