import torch
import MinkowskiEngine as ME
import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt

from coffee_nn.models.sparse import SuperPoint
from . import UntrainedCOFFEEBackend

# Celestial Occlusion Fast FEature Extractor
class TrainedCOFFEEBackend(UntrainedCOFFEEBackend):

    def __init__(self, client, server, size, backend_params):
        super().__init__(client, server, size, backend_params)

        # Setup output for model weights
        if backend_params["model_type"] != "PTH":
            raise NotImplementedError("Supported outputs only include 'PTH' Pytorch models")
        
        # Model instantiation
        model_wrapped = SuperPoint()
        self._model = model_wrapped.to(self._device)
        self._model.load_state_dict(torch.load(backend_params["model_path"]))
        self._model.eval()
    
    def detect_features(self, image):
        (in_coords, in_features) = super().detect_features(image)
        
        #mean = 1.471
        #std = 0.0452
        #transform = lambda x: (x - mean)/std

        compatible_coords = torch.hstack((torch.zeros_like(in_features, dtype=torch.int), in_coords))

        input = ME.SparseTensor(in_features, compatible_coords) # Add empty batch dimension

        with torch.set_grad_enabled(False):
            output = self._model.forward(input)

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
        
        