import torch
import MinkowskiEngine as ME
import numpy as np
import time
from matplotlib import colors
from matplotlib import pyplot as plt

from coffee_nn.hardware import GPU

from .non_trained import UntrainedCOFFEEBackend
from .descriptor import COFFEEDescriptor
from .filter import COFFEEFilter

# Celestial Occlusion Fast FEature Extractor
class TrainedCOFFEEBackend(UntrainedCOFFEEBackend):

    def __init__(self, client, server, size, backend_params, gpu=None):
        super().__init__(client, server, size, backend_params, gpu=gpu)
        self._descriptor = COFFEEDescriptor(gpu=gpu, model_path=backend_params["descriptor_model_path"])
    
    def detect_features(self, image):
        (in_coords, in_features) = super().detect_features(image)

        # Add batch coordinate to make it compatible with the Minkowski Engine
        compatible_coords = torch.hstack((torch.zeros_like(in_features, dtype=torch.int), in_coords))

        with torch.set_grad_enabled(False):            
            (filtered_coords, filtered_features) = (compatible_coords, in_features)
            
            #(x - 1.458) / 0.2087
            mu = filtered_features.mean()
            sigma = filtered_features.std()
            
            out_features = self._descriptor.forward_sparse(filtered_coords, (filtered_features - mu) / sigma)
               
        return (filtered_coords[:, 1:3].cpu(), out_features.cpu())
        
        