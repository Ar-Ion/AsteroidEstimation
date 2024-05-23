import torch
from .loss import LossFunction

class KLLoss(LossFunction):
    def __init__(self, margin, weight):
        self._margin = margin
        self._weight = weight
        
    def loss(self, true_dists, true_matches, pred_dists, pred_matches):       
        true_dustbin = torch.maximum(torch.tensor(0), 1 - true_matches.sum(dim=0))
        pred_dustbin = torch.maximum(torch.tensor(0), 1 - pred_dists.sum(dim=0))

        truth = torch.nn.functional.normalize(torch.vstack((true_matches, true_dustbin)), dim=0, p=1)
        pred = torch.nn.functional.normalize(torch.vstack((pred_dists, pred_dustbin)) + 1e-12, dim=0, p=1)
        
        sigma = 2
        num_entries = truth.shape[1]
        noise_likelihood = -2 * torch.ones((1, num_entries))
        expected = torch.log_softmax(torch.vstack((-true_dists**2/(2*sigma*sigma), noise_likelihood)), dim=0)

        kl_div = torch.sum(torch.exp(expected) * (expected - torch.log(pred)), dim=0)
        
        return kl_div.sum(), 1