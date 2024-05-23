import torch
import MinkowskiEngine as ME
import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt

from .non_trained import UntrainedCOFFEEBackend
from .descriptor import COFFEEDescriptor

# Celestial Occlusion Fast FEature Extractor
class TrainedCOFFEEBackend(UntrainedCOFFEEBackend):

    def __init__(self, client, server, size, backend_params):
        super().__init__(client, server, size, backend_params)
        self._descriptor = COFFEEDescriptor(model_path=backend_params["model_path"])
    
    def detect_features(self, image):
        (in_coords, in_features) = super().detect_features(image)

        compatible_coords = torch.hstack((torch.zeros_like(in_features, dtype=torch.int), in_coords))

        input = ME.SparseTensor(in_features, compatible_coords) # Add empty batch dimension

        with torch.set_grad_enabled(False):
            output = self._descriptor.model.forward(input)

        out_coords = output.coordinates[:, 1:3].cpu()
        out_features = output.features.cpu()

        # img = np.zeros((1024, 1024, 3))

        # mapper = np.linspace(0, 1, out_features.shape[1])
        # hue = np.dot(out_features, mapper)
        # value = np.linalg.norm(out_features, axis=1)

        # hsv = np.array((hue, np.ones_like(hue), value))
        
        # img[out_coords[:, 0], out_coords[:, 1]] = colors.hsv_to_rgb(hsv.T)

        # plt.imshow(img)
        # plt.show()
                                
        return (out_coords, out_features)
        
        