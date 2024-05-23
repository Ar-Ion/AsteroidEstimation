import torch
from .loss import LossFunction

class CrossEntropyLoss(LossFunction):
    def __init__(self):
        pass
        
    def loss(self, true_dists, true_matches, pred_dists, pred_matches):       
        # Ground truth
        
        true_dustbin = torch.maximum(torch.tensor(0), 1 - true_matches.sum(dim=0))
        pred_dustbin = torch.maximum(torch.tensor(0), 1 - pred_dists.sum(dim=0))
        
        truth = torch.nn.functional.normalize(torch.vstack((true_matches, true_dustbin)), dim=0, p=1)
        pred = torch.nn.functional.normalize(torch.vstack((pred_dists, pred_dustbin)) + 1e-12, dim=0, p=1)
        
        gain = torch.log(pred) * truth
    
        return -gain.sum(), 1