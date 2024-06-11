import torch
from coffee_nn.models.matchers import SuperGlue
from .matcher import Matcher

class SuperglueMatcher(Matcher):
    def __init__(self, model):
        self._model = model
        
    def match(self, data):

        output = self._model({
            "keypoints0": data.prev_points.kps, 
            "keypoints1": data.next_points.kps, 
            "descriptors0": data.prev_points.features,
            "descriptors1": data.next_points.features,
            "scores0": torch.ones((len(data.prev_points.features)))[:, None],
            "scores1": torch.ones((len(data.next_points.features)))[:, None],
            "image_shape": (0, 0, 1024, 1024)
        })[0, :, :]
        
        pred_dists = output[:-1, :-1].exp() # Remove dustbin and convert log-score to score
        pred_matches = self._criterion.apply(pred_dists)
                
        return pred_dists, pred_matches