import torch
import MinkowskiEngine as ME

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
        input = ME.SparseTensor(in_features[:, None], in_coords[None, :]) # Add empty batch dimension

        output = self._model.forward(input)
                    
        return (output.coordinates, output.features)
        
        