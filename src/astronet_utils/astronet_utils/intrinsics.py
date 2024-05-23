import torch

from astronet_msgs import ProjectionData
from .abstractions import Batchable, Torchable

class IntrinsicsUtils(Batchable, Torchable):
    # Constructor helpers
    def from_K(k):
        K = torch.from_numpy(k).float()
        K_inv = torch.inverse(K)
        return ProjectionData.IntrinsicsData(K, K_inv)
    
    # Functionalities
    def apply(intrinsics, point):
        # Add a dimension to point to facilitate broadcasting
        point = point[:, :, None]
        
        # First, apply the intrisic matrix
        ndc = torch.einsum("ij,...jl->...il", intrinsics.K, point)
        
        # Then, normalize the coordinates by the depth
        return torch.hstack((ndc[:, 0]/ndc[:, 2], ndc[:, 1]/ndc[:, 2], ndc[:, 2]))

    def revert(intrinsics, point):
        # Add a dimension to point to facilitate broadcasting
        point = point[:, :, None]
        
        # Let's first get the UV coordinates in the image frame (without depth)
        uv = torch.hstack((point[:, 0], point[:, 1], torch.ones(point.shape[0], 1)))

        # Now, we transform the UV coordinates in normalized device coordinates (NDC) through the intrinsic matrix K
        ndc = torch.einsum("ij,...j->...i", intrinsics.K_inv, uv)

        # Finally, we add back the depth information by normalizing the NDC coordinates
        return ndc * point[:, 2]
      
    # Compliance with Torchable abstraction
    def to(intrinsics, device=None, dtype=None):
        K = intrinsics.K.to(device=device, dtype=dtype)
        K_inv = intrinsics.K_inv.to(device=device, dtype=dtype)
        return ProjectionData.IntrinsicsData(K, K_inv)
        
    def detach(extrinsics):
        K = K.detach()
        K_inv = K_inv.detach()
        return ProjectionData.ExtrinsicsData(K, K_inv)
    
    # Compliance with Batchable abstraction
    def batched(instrinsics_data_list):
        num_batches = len(instrinsics_data_list)
        
        collated_K = [x.K for x in instrinsics_data_list]
        collated_K_inv = [x.K_inv for x in instrinsics_data_list]
        
        # Concatenate all lists
        batched_K = torch.stack(collated_K)
        batched_K_inv = torch.stack(collated_K_inv)
        
        return ProjectionData.ExtrinsicsData(batched_K, batched_K_inv, num_batches)
        
    def retrieve(intrinsics_data, index):
        K = intrinsics_data.K[index, :]
        K_inv = intrinsics_data.K_inv[index, :]
        return ProjectionData.IntrinsicsData(K, K_inv)
