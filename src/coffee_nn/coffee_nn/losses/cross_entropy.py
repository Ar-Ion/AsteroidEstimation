import torch
from .loss import LossFunction

class CrossEntropyLoss(LossFunction):
    def __init__(self):
        self._eps = 0.1
        
    def loss(self, true_dists, true_matches, pred_dists, pred_matches):       
        # Perhaps use label smoothing?
        u = torch.rand_like(true_matches)*self._eps
        true_matches_smoothed = (1-self._eps)*true_matches + self._eps*u/true_matches.shape[0]
        
        true_bins = torch.maximum(torch.tensor(0), true_matches_smoothed)
        pred_bins = torch.maximum(torch.tensor(0), pred_dists)
        true_dustbin = torch.maximum(torch.tensor(0), 1 - true_matches_smoothed.sum(dim=0))
        pred_dustbin = torch.maximum(torch.tensor(0), 1 - pred_dists.sum(dim=0))
        
        truth = torch.nn.functional.normalize(torch.vstack((true_bins, true_dustbin)), dim=0, p=1)
        pred = torch.nn.functional.normalize(torch.vstack((pred_bins, pred_dustbin)), dim=0, p=1)
        
        gain = torch.log(torch.maximum(torch.tensor(1e-12), pred)) * truth
    
        return -gain.sum(), 1