import os
import torch
import numpy as np
from torch.utils.cpp_extension import load
from ament_index_python.packages import get_package_share_directory

from coffee_nn.hardware import GPU

from .. import Backend

# Celestial Occlusion Fast FEature Extractor
class UntrainedCOFFEEBackend(Backend):

    def __init__(self, client, server, size, backend_params, gpu=None):

        super().__init__(client, server, size, backend_params)
        
        if gpu != None:
            self._gpu = gpu
        else:
            self._gpu = GPU("cuda:0")
        
        module_dir = get_package_share_directory("feature_descriptor")
        cuda_module = os.path.join(module_dir, "cuda_modules", "sparsify_cuda.cpp")
        cuda_kernel = os.path.join(module_dir, "cuda_kernels", "sparsify_cuda_kernel.cu")
         
        print("Building CUDA module and kernel...")
        self._coffee_cuda = load(name='sparsify', sources=[cuda_module, cuda_kernel])
        print("CUDA module and kernel built")
        
        if backend_params["design_param"]:
            self._threshold = int(backend_params["design_param"])
        else:
            self._threshold = 50
            
        if backend_params["max_features"]:
            self._max_features = int(backend_params["max_features"])
        else:
            self._max_features = 4096
            
        print(f"Max number of features: {self._max_features}")
        print(f"Threshold: {self._threshold}")

        self._coffee_cuda.init(self._threshold)
    
    def detect_features(self, image):        
        # Convert to GPU tensor
        gpu_image = torch.from_numpy(image).to(self._gpu.device).contiguous().short()
        
        # Preprocessing and dimension reduction
        sparse_repr = self._coffee_cuda.sparsify(gpu_image)
                
        # Arctan is a more linear representation of the size of a shadow, since it is projected as a tangent light ray
        angle_repr = torch.atan(sparse_repr)
                                
        # We need to COO representation and not the CSR
        torch_coo_repr = angle_repr.to_sparse_coo()

        coords = torch_coo_repr.indices().to(dtype=torch.int)
        features = torch_coo_repr.values()
        
        # Filter out negative angles
        coords = coords[:, features > 0]
        features = features[features > 0]
                
        # Take the "best" features
        if coords.shape[1] > self._max_features:
            most_relevant = torch.topk(features, self._max_features)
            coords = coords[:, most_relevant.indices]
            features = features[most_relevant.indices]
        
        return (coords.T, features[:, None])
        
        