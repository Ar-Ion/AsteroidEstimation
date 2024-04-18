import os
import torch
from torch.utils.cpp_extension import load
from ament_index_python.packages import get_package_share_directory

from .. import Backend

# Celestial Occlusion Fast FEature Extractor
class COFFEE_Backend(Backend):

    def __init__(self, client, server, size):

        super().__init__(client, server, size)
        
        module_dir = get_package_share_directory("feature_detector")
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
    
    def detect_features(self, image):
        # Convert to GPU tensor
        gpu_image = torch.from_numpy(image).to(self._device).short()
        
        # Preprocessing and dimension reduction
        sparse_repr = self._coffee_cuda.sparsify(gpu_image)
                
        # Arctan is a more linear representation of the size of a shadow, since it is projected as a tangent light ray
        angle_repr = torch.atan(sparse_repr)
                
        # We need to COO representation and not the CSR
        torch_coo_repr = angle_repr.to_sparse_coo()

        # Convert back to CPU tensor
        coords = torch_coo_repr.indices().cpu()
        features = torch_coo_repr.values().cpu()

        max_features = 4096

        if coords.shape[1] > max_features:
            most_relevant = torch.topk(features, max_features)
            return (coords.T[most_relevant.indices], features[most_relevant.indices])
        else:
            return (coords.T, features)