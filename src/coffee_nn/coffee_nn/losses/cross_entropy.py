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
        true_dustbin1 = torch.maximum(torch.tensor(0), 1 - true_matches_smoothed.sum(dim=0))
        true_dustbin2 = torch.maximum(torch.tensor(0), 1 - true_matches_smoothed.sum(dim=1))
        true_dustbin1 = torch.maximum(torch.tensor(0), 1 - true_matches_smoothed.sum(dim=0))
        true_dustbin2 = torch.maximum(torch.tensor(0), 1 - true_matches_smoothed.sum(dim=1))
        
        truth = torch.hstack((torch.vstack((true_bins, true_dustbin1[None, :])), torch.vstack((true_dustbin2[:, None], torch.tensor(0)))))
        truth = torch.hstack((torch.vstack((true_bins, true_dustbin1[None, :])), torch.vstack((true_dustbin2[:, None], torch.tensor(0)))))
        
        reward = pred_dists * truth
        reward = pred_dists * truth
    
        return -reward.sum(), 1 # truth.sum() Normalizing the loss breaks everything in the optimizer
