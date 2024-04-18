import os
import torch
import cv2
import MinkowskiEngine as ME
from matplotlib import pyplot as plt
from torch.utils.cpp_extension import load
from ament_index_python.packages import get_package_share_directory

from .. import Backend
from .models import SuperPoint
import cProfile

# Celestial Occlusion Fast FEature Extractor
class COFFEE_V2_Backend(Backend):

    def __init__(self):
        
        module_dir = get_package_share_directory("feature_extractor")
        cuda_module = os.path.join(module_dir, "cuda_modules", "sparsify_cuda.cpp")
        cuda_kernel = os.path.join(module_dir, "cuda_kernels", "sparsify_cuda_kernel.cu")
         
        print("Loading GPU...")
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        print("GPU loaded. Using compute device: " + str(self._device))

        print("Building CUDA module and kernel...")
        self._coffee_cuda = load(name='sparsify', sources=[cuda_module, cuda_kernel])
        print("CUDA module and kernel built")

        self._coffee_cuda.init(100)

        self._model = SuperPoint().to(self._device)
    
    def extract_features(self, image):
        # Convert to GPU tensor
        gpu_image = torch.from_numpy(image).to(self._device).short()
        
        # Preprocessing and dimension reduction

        filtered_features = []
        filtered_coords = []

        for i in range(gpu_image.shape[0]):
            sparse_repr = self._coffee_cuda.sparsify(gpu_image[i])
                    
            angle_repr = torch.atan(sparse_repr)
                    
            torch_coo_repr = angle_repr.to_sparse_coo()
            
            torch_features = torch_coo_repr.values()
            torch_coords = torch_coo_repr.indices()
            filter = torch_features > 0 # Only consider shadow start
            torch_f_features = torch_features[filter]
            torch_f_coords = torch_coords[:, filter]

            #torch_f_features = torch.ones_like(torch_f_features)

            filtered_coords.append(torch_f_coords.T)
            filtered_features.append(torch_f_features[:, None])

        coords, features = ME.utils.sparse_collate(filtered_coords, filtered_features)
        
        me_coo_in = ME.SparseTensor(features=features, coordinates=coords)
        me_coo_out = self._model.forward(me_coo_in)

        coords = me_coo_out.coordinates
        features = torch.nn.functional.normalize(me_coo_out.features)

        return (coords, features)
                
    def get_match_norm(self):
        return cv2.NORM_L2

    def get_model(self):
        return self._model