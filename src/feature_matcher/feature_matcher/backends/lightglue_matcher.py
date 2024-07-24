import torch

from coffee_nn.hardware import GPU
from coffee_nn.common import Model
from coffee_nn.models.matchers import LightGlue

from .matcher import Matcher

class LightglueMatcher(Model, Matcher):
    def __init__(self, model_path, criterion=None, autoload=True, gpu=None):
        Matcher.__init__(self, criterion)
        Model.__init__(self, gpu, LightGlue(features="coffee").float(), model_path, autoload=autoload)

    def match(self, data):
        
        size = torch.tensor(((1024, 1024)))

        a_data = {
            "keypoints": data.prev_points.kps[None, :, :], 
            "descriptors": data.prev_points.features[None, :], 
            "image_size": size
        }
        
        b_data = {
            "keypoints": data.next_points.kps[None, :, :], 
            "descriptors": data.next_points.features[None, :], 
            "image_size": size
        }

        output = self.model({
            "image0": a_data, 
            "image1": b_data
        })
        
        scores = output["scores"][0]
        
        pred_dists = scores # Remove dustbin and convert log-score to score
        pred_matches = self._criterion.apply(pred_dists[:-1, :-1].exp())
                
        return pred_dists, pred_matches