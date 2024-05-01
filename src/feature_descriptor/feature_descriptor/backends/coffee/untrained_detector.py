import os
import torch
from torch.utils.cpp_extension import load
from ament_index_python.packages import get_package_share_directory
from matplotlib import pyplot as plt

from .. import Backend

# Celestial Occlusion Fast FEature Extractor
class UntrainedCOFFEEBackend(Backend):

    def __init__(self, client, server, size, backend_params):

        super().__init__(client, server, size, backend_params)
        
        module_dir = get_package_share_directory("feature_descriptor")
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
        #plt.figure()
        #plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        
        # Convert to GPU tensor
        gpu_image = torch.from_numpy(image).to(self._device).short()
        
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
        
        # Take the 4096 "best" features
        max_features = 4096

        if coords.shape[1] > max_features:
            most_relevant = torch.topk(features, max_features)
            coords = coords[:, most_relevant.indices]
            features = features[most_relevant.indices]
            
        #plt.figure()
        #plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        #plt.scatter(coords[1, :], coords[0, :], s=0.1)
        #plt.xlim(0, 1024)
        #plt.ylim(1024, 0)
        #plt.show()
        
        return (coords.T, features)
        
        