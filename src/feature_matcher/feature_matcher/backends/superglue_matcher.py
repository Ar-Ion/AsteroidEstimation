import torch
from coffee_nn.models.matchers import SuperGlue
from .matcher import Matcher

class SuperglueMatcher(Matcher):
    def __init__(self, model):
        self._model = model
        
    def match(self, data):

        output = self._model({
            "keypoints0": data.prev_kps, 
            "keypoints1": data.next_kps, 
            "descriptors0": data.prev_features,
            "descriptors1": data.next_features,
            "scores0": torch.ones((len(data.prev_features)))[:, None],
            "scores1": torch.ones((len(data.next_features)))[:, None],
            "image_shape": (0, 0, 1024, 1024)
        })[0, :, :]
        
        pred_dists = output[:-1, :-1].exp() # Remove dustbin and convert log-score to score
        pred_matches = self._criterion.apply(pred_dists)
                
        return pred_dists, pred_matches