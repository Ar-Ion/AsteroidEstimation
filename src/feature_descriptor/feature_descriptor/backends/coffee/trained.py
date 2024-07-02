import torch
import MinkowskiEngine as ME
import numpy as np
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
        self._filter = COFFEEFilter(gpu=gpu, model_path=backend_params["filter_model_path"])
        self._descriptor = COFFEEDescriptor(gpu=gpu, model_path=backend_params["descriptor_model_path"])
    
    def detect_features(self, image):
        (in_coords, in_features) = super().detect_features(image)

        # Add batch coordinate to make it compatible with the Minkowski Engine
        compatible_coords = torch.hstack((torch.zeros_like(in_features, dtype=torch.int), in_coords))

        with torch.set_grad_enabled(False):
            in_count = len(in_features)
            (filtered_coords, filtered_features) = (compatible_coords, in_features)
            out_features = self._descriptor.forward_sparse(filtered_coords, filtered_features)
            
            print(len(out_features)/in_count)

        # img = np.zeros((1024, 1024, 3))

        # mapper = np.linspace(0, 1, out_features.shape[1])
        # hue = np.dot(out_features.cpu(), mapper)/3
        # value = np.linalg.norm(out_features.cpu(), axis=1)

        # hsv = np.array((hue, np.ones_like(hue), value))
        
        # img[filtered_coords[:, 1].cpu(), filtered_coords[:, 2].cpu()] = colors.hsv_to_rgb(hsv.T)

        # plt.imshow(img)
        # plt.pause(0.1)
                                
        return (filtered_coords[:, 1:3].cpu(), out_features.cpu())
        
        